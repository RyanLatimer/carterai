//! Per-URL system/PAC proxy resolution, explicit redirects, and credential-safe diagnostics.
use crate::settings::Settings;
use anyhow::{Context, Result, bail};
use reqwest::{
    Url,
    blocking::{Client, Response},
    header,
};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    sync::{Mutex, OnceLock},
    time::Duration,
};

pub fn diagnostics_path() -> Result<std::path::PathBuf> {
    Ok(crate::settings::data_dir()?.join("network.log"))
}
static LOG: OnceLock<Mutex<()>> = OnceLock::new();
pub fn log(line: &str) {
    let _guard = LOG.get_or_init(|| Mutex::new(())).lock().ok();
    let Ok(path) = diagnostics_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    if path.metadata().is_ok_and(|m| m.len() > 1_000_000) {
        let _ = fs::rename(&path, path.with_extension("previous.log"));
    }
    let mut options = fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    if let Ok(mut file) = options.open(path) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let _ = writeln!(file, "{now} {line}");
    }
}
pub fn safe_url(url: &Url) -> String {
    let mut u = url.clone();
    let _ = u.set_username("");
    let _ = u.set_password(None);
    u.set_query(None);
    u.set_fragment(None);
    u.to_string()
}
pub fn record_error(error: &anyhow::Error) {
    // HTTP errors have their URLs removed at the boundary. Request URLs and routes are
    // logged separately without credentials or signed query strings.
    log(&format!("ERROR {error:#?}"));
}
pub fn parse_proxy(value: &str) -> Result<Url> {
    let url =
        Url::parse(value).context("Proxy must be a URL, for example http://127.0.0.1:18081")?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https" | "socks5" | "socks5h") && url.host_str().is_some(),
        "Proxy must use HTTP, HTTPS, SOCKS5, or SOCKS5H"
    );
    Ok(url)
}
fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .or_else(|| std::env::var(name.to_lowercase()).ok())
}
fn bypass(url: &Url, exclusions: &str) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_matches(['[', ']']).to_lowercase();
    exclusions
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .any(|rule| {
            if rule == "*" {
                return true;
            }
            if let (Ok(network), Ok(address)) = (
                rule.parse::<ipnet::IpNet>(),
                host.parse::<std::net::IpAddr>(),
            ) {
                return network.contains(&address);
            }
            // Optional port suffix for host/domain rules. Bracketed IPv6 remains intact.
            let (name, port) = if let Some((name, port)) = rule.rsplit_once(':') {
                if !name.contains(':') && port.parse::<u16>().is_ok() {
                    (name, port.parse::<u16>().ok())
                } else {
                    (rule, None)
                }
            } else {
                (rule, None)
            };
            if port.is_some() && port != url.port_or_known_default() {
                return false;
            }
            let name = name
                .trim_start_matches("*.")
                .trim_start_matches('.')
                .trim_matches(['[', ']'])
                .to_lowercase();
            host == name || host.ends_with(&format!(".{name}"))
        })
}
fn route(url: &Url, settings: &Settings) -> Result<Option<Url>> {
    let exceptions = env("NO_PROXY").unwrap_or_else(|| settings.no_proxy.clone());
    if bypass(url, &exceptions) {
        return Ok(None);
    }
    let specific = if url.scheme() == "https" {
        env("HTTPS_PROXY")
    } else {
        env("HTTP_PROXY")
    };
    let explicit = specific.or_else(|| env("ALL_PROXY"));
    if let Some(p) = explicit {
        return if p.trim().is_empty() {
            Ok(None)
        } else {
            Ok(Some(parse_proxy(p.trim())?))
        };
    }
    let manual = if url.scheme() == "https" {
        &settings.https_proxy
    } else {
        &settings.http_proxy
    };
    if !manual.trim().is_empty() {
        return Ok(Some(parse_proxy(manual.trim())?));
    }
    system_route(url)
}
#[cfg(not(target_os = "macos"))]
fn system_route(_: &Url) -> Result<Option<Url>> {
    Ok(None)
}
#[cfg(target_os = "macos")]
fn system_route(url: &Url) -> Result<Option<Url>> {
    use std::ffi::{CStr, CString, c_char, c_int};
    unsafe extern "C" {
        fn carter_proxy_for_url(url: *const c_char, out: *mut c_char, capacity: usize) -> c_int;
    }
    let input = CString::new(url.as_str())?;
    let mut out = [0 as c_char; 4096];
    // The C shim bounds every write and returns a NUL-terminated route or diagnostic.
    let result = unsafe { carter_proxy_for_url(input.as_ptr(), out.as_mut_ptr(), out.len()) };
    let output = unsafe { CStr::from_ptr(out.as_ptr()) }.to_string_lossy();
    match result {
        0 => Ok(None),
        1 => Ok(Some(parse_proxy(&output)?)),
        _ => bail!("{output}"),
    }
}

pub struct Http {
    settings: Settings,
    clients: HashMap<String, Client>,
}
impl Http {
    pub fn new(settings: &Settings) -> Result<Self> {
        settings.validate()?;
        Ok(Self {
            settings: settings.clone(),
            clients: HashMap::new(),
        })
    }
    pub fn get(
        &mut self,
        url: &str,
        range: Option<(u64, Option<u64>)>,
        etag: Option<&str>,
    ) -> Result<Response> {
        anyhow::ensure!(
            !self.settings.offline(),
            "Offline mode is enabled; import a model folder or use a cached model"
        );
        let mut url = Url::parse(url)?;
        let origin = Url::parse(&self.settings.endpoint()?)?.origin();
        for hop in 0..=10 {
            let proxy = route(&url, &self.settings)
                .with_context(|| format!("Proxy resolution failed for {}", safe_url(&url)))?;
            let key = proxy.as_ref().map(Url::to_string).unwrap_or_default();
            log(&format!(
                "GET {} proxy={} hop={hop}",
                safe_url(&url),
                proxy
                    .as_ref()
                    .map(safe_url)
                    .unwrap_or_else(|| "DIRECT".into())
            ));
            if !self.clients.contains_key(&key) {
                if self.clients.len() >= 16 {
                    self.clients.clear();
                }
                let builder = Client::builder()
                    .no_proxy()
                    .redirect(reqwest::redirect::Policy::none())
                    .connect_timeout(Duration::from_secs(20))
                    .timeout(Duration::from_secs(30))
                    .user_agent(concat!("Carter/", env!("CARGO_PKG_VERSION")));
                #[cfg(target_os = "macos")]
                let builder = builder.use_native_tls();
                let builder = if let Some(proxy) = proxy {
                    builder.proxy(reqwest::Proxy::all(proxy)?)
                } else {
                    builder
                };
                self.clients.insert(key.clone(), builder.build()?);
            }
            let mut request = self.clients[&key].get(url.clone());
            if let Some((start, end)) = range {
                request = request.header(
                    header::RANGE,
                    format!(
                        "bytes={start}-{}",
                        end.map(|n| n.to_string()).unwrap_or_default()
                    ),
                );
            }
            if let Some(etag) = etag {
                request = request.header(header::IF_RANGE, etag);
            }
            if url.origin() == origin
                && let Some(token) = self.settings.token()
            {
                anyhow::ensure!(
                    url.scheme() == "https"
                        || url
                            .host_str()
                            .is_some_and(|h| matches!(h, "localhost" | "127.0.0.1" | "[::1]")),
                    "HF_TOKEN requires an HTTPS endpoint"
                );
                request = request.bearer_auth(token);
            }
            let response = request.send().map_err(reqwest::Error::without_url)
                .with_context(|| format!("Connection to {} failed. Check Network settings/proxy; see network.log for the full cause",url.host_str().unwrap_or("endpoint")))?;
            let code = response.status();
            if matches!(code.as_u16(), 301 | 302 | 303 | 307 | 308) {
                let location = response
                    .headers()
                    .get(header::LOCATION)
                    .context("Redirect had no Location header")?
                    .to_str()?;
                let next = url.join(location)?;
                anyhow::ensure!(
                    next.scheme() == "https" || (url.scheme() == "http" && next.scheme() == "http"),
                    "Refusing HTTPS downgrade redirect"
                );
                log(&format!(
                    "HTTP {code} {} -> {}",
                    safe_url(&url),
                    safe_url(&next)
                ));
                url = next;
                continue;
            }
            log(&format!("HTTP {code} {}", safe_url(&url)));
            if matches!(code.as_u16(), 401 | 403) {
                bail!(
                    "HTTP {code}: access denied by endpoint or proxy. Check Network settings and HF_TOKEN."
                );
            }
            return response
                .error_for_status()
                .map_err(reqwest::Error::without_url)
                .context("Model server returned an HTTP error");
        }
        bail!("Model endpoint exceeded 10 redirects; see network.log")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proxy_exceptions_and_redaction() {
        let u = Url::parse("https://models.example.org:443/file?token=secret").unwrap();
        assert!(bypass(&u, "localhost,.example.org:443"));
        assert!(!bypass(&u, ".example.org:80,other.example"));
        assert!(bypass(
            &Url::parse("http://10.2.3.4").unwrap(),
            "10.0.0.0/8"
        ));
        assert_eq!(
            safe_url(&Url::parse("http://u:secret@proxy.example:99/a?secret=x").unwrap()),
            "http://proxy.example:99/a"
        );
        assert!(!safe_url(&u).contains("secret"));
    }
}
