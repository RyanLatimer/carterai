use crate::{State, settings::Settings};
use rune::{El, TextEdit, TextStyle, col, row, text};
use rune_kit::{Button, ButtonVariant, Kit, TextInput};

pub struct Preferences {
    pub open: bool,
    pub settings: Settings,
    http: TextEdit,
    https: TextEdit,
    bypass: TextEdit,
    endpoint: TextEdit,
    token: TextEdit,
    folder: TextEdit,
}
impl Default for Preferences {
    fn default() -> Self {
        let settings = Settings::load();
        Self {
            open: false,
            http: TextEdit::new(&settings.http_proxy),
            https: TextEdit::new(&settings.https_proxy),
            bypass: TextEdit::new(&settings.no_proxy),
            endpoint: TextEdit::new(&settings.endpoint),
            token: TextEdit::new(&settings.token),
            folder: TextEdit::new(&settings.import_folder),
            settings,
        }
    }
}
impl Preferences {
    fn save(&mut self) -> anyhow::Result<()> {
        let mut settings = self.settings.clone();
        settings.http_proxy = self.http.text.trim().into();
        settings.https_proxy = self.https.text.trim().into();
        settings.no_proxy = self.bypass.text.trim().into();
        settings.endpoint = self.endpoint.text.trim().into();
        settings.token = self.token.text.trim().into();
        settings.import_folder = self.folder.text.trim().into();
        settings.save()?;
        self.settings = settings;
        self.open = false;
        Ok(())
    }
}
pub fn view(s: &State, k: &Kit) -> El {
    let p = &s.preferences;
    let caption = TextStyle::new(12.0, 400.0).lh(18.0);
    let mut panel = col().key("preferences").w_full().gap(12.0).padding(18.0).radius(12.0).fill(k.c.layer_02)
        .child(text("Network & models").font(TextStyle::new(16.0,500.0)).color(k.c.text_base))
        .child(text("Automatic: macOS system proxies and PAC. Environment variables override saved settings.").wrap_anywhere().font(caption).color(k.c.text_muted));
    macro_rules! field {
        ($label:expr,$name:ident,$placeholder:expr) => {
            panel = panel.child(
                col()
                    .w_full()
                    .gap(4.0)
                    .child(text($label).font(caption).color(k.c.text_muted))
                    .child(
                        TextInput::new(&p.$name)
                            .placeholder($placeholder)
                            .width(None)
                            .on_edit(|s: &mut State| &mut s.preferences.$name)
                            .build(k),
                    ),
            );
        };
    }
    field!("HTTP proxy", http, "Automatic, or http://127.0.0.1:18081");
    field!("HTTPS proxy", https, "Automatic, or http://127.0.0.1:18081");
    field!(
        "Bypass proxy (NO_PROXY)",
        bypass,
        "localhost,127.0.0.1,.example.org"
    );
    field!(
        "Hugging Face endpoint (HF_ENDPOINT)",
        endpoint,
        "https://huggingface.co"
    );
    panel = panel.child(
        col()
            .w_full()
            .gap(4.0)
            .child(
                text("Hugging Face token (saved in macOS Keychain)")
                    .font(caption)
                    .color(k.c.text_muted),
            )
            .child(
                TextInput::new(&p.token)
                    .secure()
                    .placeholder("Optional HF_TOKEN")
                    .width(None)
                    .on_edit(|s: &mut State| &mut s.preferences.token)
                    .build(k),
            ),
    );
    field!(
        "Local model folder (config.json, tokenizer.json, model.safetensors)",
        folder,
        "Optional folder path; clear to download models"
    );
    let backend = if p.settings.backend == "auto" {
        "Automatic"
    } else {
        &p.settings.backend
    };
    panel = panel.child(
        row()
            .w_full()
            .gap(10.0)
            .child(
                Button::new("Choose folder…")
                    .disabled(s.busy)
                    .on_press(|s: &mut State| {
                        #[cfg(target_os = "macos")]
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("Choose GPT-2 model folder")
                            .pick_folder()
                        {
                            s.preferences.folder = TextEdit::new(path.to_string_lossy());
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            s.status = "Enter the model folder path above".into();
                        }
                    })
                    .build(k),
            )
            .child(
                Button::new(if p.settings.offline {
                    "Offline: on"
                } else {
                    "Offline: off"
                })
                .on_press(|s: &mut State| {
                    s.preferences.settings.offline = !s.preferences.settings.offline
                })
                .build(k),
            )
            .child(
                Button::new(format!("Compute: {backend}"))
                    .on_press(|s: &mut State| {
                        let backend = &mut s.preferences.settings.backend;
                        *backend = match backend.as_str() {
                            "auto" => "candle",
                            "candle" => "mlx",
                            _ => "auto",
                        }
                        .into();
                    })
                    .build(k),
            ),
    );
    panel = panel.child(
        row()
            .w_full()
            .gap(10.0)
            .child(
                Button::new("Save & load model")
                    .disabled(s.busy)
                    .on_press_tx(|s: &mut State, tx| {
                        tx.instant();
                        match s.preferences.save() {
                            Ok(()) => {
                                s.error = None;
                                s.load(s.selected);
                            }
                            Err(e) => {
                                crate::network::record_error(&e);
                                s.error = Some(format!("{e:#}"));
                                s.status = "Could not save settings".into();
                            }
                        }
                    })
                    .build(k),
            )
            .child(
                Button::new("Close")
                    .variant(ButtonVariant::Ghost)
                    .on_press(|s: &mut State| s.preferences.open = false)
                    .build(k),
            )
            .child(
                Button::new("Open diagnostics")
                    .variant(ButtonVariant::GhostMuted)
                    .on_press(|_s: &mut State| {
                        if let Ok(path) = crate::network::diagnostics_path() {
                            #[cfg(target_os = "macos")]
                            {
                                let _ = std::process::Command::new("open").arg(path).spawn();
                            }
                            #[cfg(not(target_os = "macos"))]
                            {
                                _s.status = path.display().to_string();
                            }
                        }
                    })
                    .build(k),
            ),
    );
    if let Some(error) = &s.error {
        panel = panel.child(
            text(error)
                .w_full()
                .wrap_anywhere()
                .font(caption)
                .color(k.c.danger.fg),
        );
    }
    panel
}
