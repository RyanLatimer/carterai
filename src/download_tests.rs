use super::*;
use std::net::TcpListener;

fn server(status: &str, headers: &str, body: &[u8]) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let response = format!("HTTP/1.1 {status}\r\nConnection: close\r\n{headers}\r\n").into_bytes();
    let body = body.to_vec();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut b = [0; 1];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut b).unwrap();
            request.push(b[0]);
        }
        stream.write_all(&response).unwrap();
        stream.write_all(&body).unwrap();
        String::from_utf8(request).unwrap()
    });
    (format!("http://{address}/model.safetensors"), handle)
}
fn temp() -> PathBuf {
    static SEQUENCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "carter-download-test-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}
#[test]
fn resumed_weights_require_correct_range_and_checksum() {
    let dir = temp();
    let part = dir.join("weights.incomplete");
    let etag = dir.join("weights.etag");
    let body = b"this is a complete model weight fixture";
    let hash = format!("{:x}", Sha256::digest(body));
    fs::write(&part, &body[..10]).unwrap();
    fs::write(&etag, "\"version-1\"").unwrap();
    let (url, worker) = server(
        "206 Partial Content",
        &format!(
            "Content-Length: {}\r\nContent-Range: bytes 10-{}/{}\r\nETag: \"version-1\"\r\n",
            body.len() - 10,
            body.len() - 1,
            body.len()
        ),
        &body[10..],
    );
    let settings = crate::settings::Settings {
        endpoint: url.clone(),
        no_proxy: "*".into(),
        ..Default::default()
    };
    let mut http = crate::network::Http::new(&settings).unwrap();
    download_file(
        &mut http,
        &url,
        &part,
        &etag,
        Some((body.len() as u64, &hash)),
        &Arc::new(AtomicBool::new(false)),
        |_, _| {},
    )
    .unwrap();
    let request = worker.join().unwrap().to_lowercase();
    assert!(request.contains("range: bytes=10-"));
    assert!(request.contains("if-range: \"version-1\""));
    assert_eq!(fs::read(&part).unwrap(), body);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn server_ignoring_range_restarts_instead_of_appending() {
    let dir = temp();
    let part = dir.join("weights.incomplete");
    let etag = dir.join("weights.etag");
    let body = b"complete new response";
    let hash = format!("{:x}", Sha256::digest(body));
    fs::write(&part, b"old").unwrap();
    let (url, worker) = server(
        "200 OK",
        &format!("Content-Length: {}\r\n", body.len()),
        body,
    );
    let settings = crate::settings::Settings {
        endpoint: url.clone(),
        no_proxy: "*".into(),
        ..Default::default()
    };
    download_file(
        &mut crate::network::Http::new(&settings).unwrap(),
        &url,
        &part,
        &etag,
        Some((body.len() as u64, &hash)),
        &Arc::new(AtomicBool::new(false)),
        |_, _| {},
    )
    .unwrap();
    worker.join().unwrap();
    assert_eq!(fs::read(&part).unwrap(), body);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn interrupted_response_preserves_partial_and_offline_makes_no_request() {
    let dir = temp();
    let part = dir.join("weights.incomplete");
    let etag = dir.join("weights.etag");
    let (url, worker) = server("200 OK", "Content-Length: 100\r\n", b"first bytes");
    let settings = crate::settings::Settings {
        endpoint: url.clone(),
        no_proxy: "*".into(),
        ..Default::default()
    };
    assert!(
        download_file(
            &mut crate::network::Http::new(&settings).unwrap(),
            &url,
            &part,
            &etag,
            None,
            &Arc::new(AtomicBool::new(false)),
            |_, _| {}
        )
        .is_err()
    );
    worker.join().unwrap();
    assert_eq!(fs::read(&part).unwrap(), b"first bytes");
    let settings = crate::settings::Settings {
        offline: true,
        ..settings
    };
    assert!(
        crate::network::Http::new(&settings)
            .unwrap()
            .get(&url, None, None)
            .unwrap_err()
            .to_string()
            .contains("Offline")
    );
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn resume_range_validation() {
    assert_eq!(parse_content_range("bytes 10-19/20").unwrap(), (10, 20));
    for invalid in [
        "bytes 10-20/20",
        "bytes 19-10/20",
        "items 1-2/3",
        "bytes 1-2/*",
    ] {
        assert!(parse_content_range(invalid).is_err());
    }
}
