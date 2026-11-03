mod engine;
mod gpt2;
mod models;

use engine::{Command, Event, Worker};
use models::{Cancel, MODELS};
use rune::{AppSpec, Cx, El, ScrollSpec, TextEdit, TextStyle, Tx, col, el, row, text};
use rune_kit::{Button, ButtonVariant, TextInput};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Message {
    user: bool,
    text: String,
}
struct State {
    input: TextEdit,
    messages: Vec<Message>,
    worker: Option<Worker>,
    selected: usize,
    ready: bool,
    busy: bool,
    generating: bool,
    status: String,
    cancel: Cancel,
    menu: bool,
    failed: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            input: TextEdit::new(""),
            messages: vec![],
            worker: None,
            selected: 0,
            ready: false,
            busy: false,
            generating: false,
            status: "Choose a model to begin".into(),
            cancel: Arc::new(AtomicBool::new(false)),
            menu: false,
            failed: false,
        }
    }
}
impl State {
    fn load(&mut self, index: usize) {
        if self.busy {
            return;
        }
        self.selected = index;
        self.ready = false;
        self.busy = true;
        self.menu = false;
        self.failed = false;
        self.cancel = Arc::new(AtomicBool::new(false));
        self.status = format!("Preparing {}…", MODELS[index].name);
        if let Some(w) = &self.worker {
            let _ = w.tx.send(Command::Load(index, self.cancel.clone()));
        }
    }
    fn send(&mut self) {
        let value = self.input.text.trim().to_string();
        if value.is_empty() || !self.ready || self.busy {
            return;
        }
        self.messages.push(Message {
            user: true,
            text: value,
        });
        let mut prompt = "A conversation between Human and Assistant.\n".to_string();
        for m in &self.messages {
            prompt.push_str(if m.user { "Human: " } else { "Assistant: " });
            prompt.push_str(&m.text);
            prompt.push('\n');
        }
        prompt.push_str("Assistant:");
        self.messages.push(Message {
            user: false,
            text: String::new(),
        });
        self.input = TextEdit::new("");
        self.busy = true;
        self.generating = true;
        self.failed = false;
        self.cancel = Arc::new(AtomicBool::new(false));
        self.status = "Thinking locally…".into();
        if let Some(w) = &self.worker {
            let _ = w.tx.send(Command::Generate(prompt, self.cancel.clone()));
        }
    }
}
fn setup(s: &mut State, wake: rune::Waker) {
    s.worker = Some(engine::start(wake));
    s.load(0);
}
fn poll(s: &mut State, _: f64, _: &mut Tx) -> bool {
    let events: Vec<_> = s
        .worker
        .as_ref()
        .map(|w| w.rx.try_iter().collect())
        .unwrap_or_default();
    let changed = !events.is_empty();
    for e in events {
        match e {
            Event::Status(v) => s.status = v,
            Event::Ready(index, backend) => {
                s.selected = index;
                s.ready = true;
                s.busy = false;
                s.status = backend;
            }
            Event::Text(v) => {
                if let Some(m) = s.messages.last_mut() {
                    m.text = v;
                }
            }
            Event::Done(v) => {
                s.busy = false;
                s.generating = false;
                s.status = v;
            }
            Event::Error(v) => {
                s.busy = false;
                s.generating = false;
                s.failed = true;
                s.status = if s.cancel.load(Ordering::Relaxed) {
                    "Stopped".into()
                } else {
                    v
                };
            }
        }
    }
    changed
}
fn view(s: &State, cx: &mut Cx<State>) -> El {
    let k = rune_theme_opencode::kit(cx.system_dark());
    let c = &k.c;
    let title = TextStyle::new(26.0, 500.0).lh(34.0);
    let body = TextStyle::new(15.0, 400.0).lh(24.0);
    let caption = TextStyle::new(12.0, 400.0).lh(18.0);
    let header = row()
        .w_full()
        .center_y()
        .gap(12.0)
        .child(
            text("carter")
                .font(TextStyle::new(18.0, 600.0))
                .color(c.text_base),
        )
        .child(el().grow(1.0))
        .child(
            Button::new(format!("{}  ⌄", MODELS[s.selected].name))
                .variant(ButtonVariant::Ghost)
                .disabled(s.busy)
                .on_press(|s: &mut State| s.menu = !s.menu)
                .build(k),
        )
        .child(
            Button::new("New chat")
                .variant(ButtonVariant::GhostMuted)
                .disabled(s.busy)
                .on_press(|s: &mut State| s.messages.clear())
                .build(k),
        );
    let mut content = col().w_full().gap(24.0);
    if s.menu {
        let mut choices = col()
            .w_full()
            .padding(16.0)
            .gap(12.0)
            .radius(12.0)
            .fill(c.layer_02);
        for (i, m) in MODELS.iter().enumerate() {
            choices = choices.child(
                row()
                    .w_full()
                    .center_y()
                    .gap(12.0)
                    .child(
                        col()
                            .grow(1.0)
                            .gap(4.0)
                            .child(text(m.name).font(body).color(c.text_base))
                            .child(text(m.detail).font(caption).color(c.text_muted)),
                    )
                    .child(
                        Button::new(if i == s.selected && s.ready {
                            "Reload"
                        } else {
                            "Use model"
                        })
                        .on_press(move |s: &mut State| s.load(i))
                        .build(k),
                    ),
            );
        }
        content = content.child(choices);
    }
    if s.messages.is_empty() {
        content = content.child(
            col()
                .w_full()
                .grow(1.0)
                .center()
                .gap(14.0)
                .child(
                    text("A little less intelligent.")
                        .font(title)
                        .color(c.text_base),
                )
                .child(
                    text("A quiet space to talk to GPT-2.\nOn your Mac. Entirely yours.")
                        .font(body)
                        .color(c.text_muted),
                )
                .child(
                    text("GPT-2 completes text. Expect odd answers.")
                        .font(caption)
                        .color(c.text_faint),
                ),
        );
    } else {
        for (i, m) in s.messages.iter().enumerate() {
            content = content.child(
                col()
                    .key(format!("message-{i}"))
                    .w_full()
                    .gap(6.0)
                    .child(
                        text(if m.user {
                            "You"
                        } else {
                            MODELS[s.selected].name
                        })
                        .font(caption)
                        .color(c.text_muted),
                    )
                    .child(
                        text(if m.text.is_empty() { "…" } else { &m.text })
                            .font(body)
                            .color(c.text_base),
                    ),
            );
        }
    }
    let action = if s.busy {
        Button::new("Stop")
            .variant(ButtonVariant::Outline)
            .on_press(|s: &mut State| {
                s.cancel.store(true, Ordering::Relaxed);
                s.status = "Stopping…".into();
            })
            .build(k)
    } else if !s.ready || (s.failed && !s.generating) {
        Button::new("Retry")
            .on_press(|s: &mut State| s.load(s.selected))
            .build(k)
    } else {
        Button::new("Send ↑")
            .variant(ButtonVariant::Submit)
            .disabled(s.input.text.trim().is_empty())
            .on_press(State::send)
            .build(k)
    };
    let composer = row()
        .w_full()
        .center_y()
        .gap(12.0)
        .padding(12.0)
        .radius(14.0)
        .fill(c.layer_02)
        .outline(1.0, 0.0, c.border_muted)
        .child(
            TextInput::new(&s.input)
                .placeholder("Say something…")
                .plain()
                .width(None)
                .height(40.0)
                .text_style(body)
                .disabled(!s.ready || s.busy)
                .on_edit(|s: &mut State| &mut s.input)
                .on_submit(|s: &mut State, _| s.send())
                .build(k)
                .grow(1.0),
        )
        .child(action);
    col()
        .w_full()
        .h_full()
        .fill(c.page)
        .padding(28.0)
        .gap(22.0)
        .child(header)
        .child(
            content
                .grow(1.0)
                .min_h(0.0)
                .scroll(ScrollSpec { stick_bottom: true }),
        )
        .child(
            col().w_full().gap(10.0).child(composer).child(
                text(format!("●  {}", s.status))
                    .font(caption)
                    .color(if s.failed { c.danger.fg } else { c.text_muted }),
            ),
        )
}
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--smoke") {
        let path = std::env::args()
            .nth(2)
            .expect("--smoke needs a model directory");
        if let Err(e) = engine::smoke(std::path::Path::new(&path)) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return;
    }
    #[cfg(target_os = "macos")]
    {
        extern crate accelerate_src;
    }
    let mut spec = AppSpec::new("Carter", (820.0, 700.0), State::default, view);
    spec.fonts = rune_theme_opencode::FONTS;
    spec.families = rune_theme_opencode::FAMILIES;
    spec.min_size = Some((560.0, 440.0));
    spec.setup = Some(setup);
    spec.poll = Some(poll);
    rune::run(spec);
}
