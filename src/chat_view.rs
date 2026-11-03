//! Streaming prose changes in place. It must not replay mount fades or spring reflow.
use rune::{El, ScrollSpec, Spring, TextStyle, col, memo, text};
use rune_kit::Kit;

pub fn empty(kit: &Kit) -> El {
    col()
        .key("empty-state")
        .w_full()
        .grow(1.0)
        .center()
        .instant_changes()
        .child(
            text("A little less intelligent.")
                .key("empty-headline")
                .font(TextStyle::new(26.0, 500.0).lh(34.0))
                .color(kit.c.text_base),
        )
}

pub fn message(index: usize, user: bool, body: &str, assistant: &str, kit: &Kit) -> El {
    // A new token only rebuilds the live reply. Completed turns reuse their keyed subtree,
    // including Rune's measured text/glyph caches, while still responding to theme changes.
    memo(format!("message-{index}"), (user, body, assistant), || {
        col()
            .key(format!("message-{index}"))
            .w_full()
            .min_w(0.0)
            .gap(6.0)
            .instant_changes()
            .child(
                text(if user { "You" } else { assistant })
                    .key(format!("message-{index}-author"))
                    .w_full()
                    .font(TextStyle::new(12.0, 400.0).lh(18.0))
                    .color(kit.c.text_muted),
            )
            .child(
                text(if body.is_empty() { "…" } else { body })
                    .key(format!("message-{index}-body"))
                    .wrap_anywhere()
                    .w_full()
                    .min_w(0.0)
                    .min_h(24.0)
                    .font(TextStyle::new(15.0, 400.0).lh(24.0))
                    .color(kit.c.text_base),
            )
    })
}

pub fn viewport(content: El) -> El {
    content
        .key("transcript-viewport")
        .w_full()
        .min_w(0.0)
        .grow(1.0)
        .min_h(0.0)
        .instant_changes()
        .motion(Spring::NONE)
        .scroll(ScrollSpec { stick_bottom: true })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rune::{Cx, Env, Runtime};

    #[derive(Default)]
    struct Transcript {
        answer: String,
        empty: bool,
    }
    fn view(state: &Transcript, _: &mut Cx<Transcript>) -> El {
        let k = rune_theme_opencode::kit(true);
        let content = if state.empty {
            col().child(empty(k))
        } else {
            col()
                .gap(24.0)
                .child(message(0, true, "Tell me something.", "Dumb", k))
                .child(message(1, false, &state.answer, "Dumb", k))
        };
        col().w_full().h_full().child(viewport(content))
    }
    fn runtime(state: Transcript) -> Runtime<Transcript> {
        let mut rt = Runtime::new(state, view, (320.0, 160.0), Env::new(true, 2.0));
        rt.frame(0.0);
        rt
    }
    #[test]
    fn wrapped_stream_updates_are_opaque_at_their_final_layout_in_the_same_frame() {
        let mut rt = runtime(Transcript {
            answer: "One".into(),
            empty: false,
        });
        let first = rt.presented("message-1-body").unwrap().0;
        for words in 1..25 {
            rt.update(|state, tx| {
                state.answer = "One more quiet sentence. ".repeat(words);
                tx.instant();
            });
            rt.frame(1.0 / 60.0);
            let body = rt
                .elements()
                .into_iter()
                .find(|e| e.key.as_deref() == Some("message-1-body"))
                .unwrap();
            assert_eq!(
                body.opacity, 1.0,
                "existing glyphs must not fade on an update"
            );
            assert_eq!(
                &body.rect[2..],
                &body.target[2..],
                "text must not spring through intermediate wrapping"
            );
            assert_eq!(body.rect[0], first[0]);
            assert_eq!(body.rect[2], first[2]);
            assert!(
                !body.moving,
                "streamed text must be at rest after its update"
            );
            let landed = body.rect;
            rt.frame(1.0 / 60.0);
            assert_eq!(
                rt.presented("message-1-body").unwrap().0,
                landed,
                "a token must not leave a delayed text/scroll animation"
            );
        }
    }
    #[test]
    fn stream_update_does_not_pull_a_reader_back_to_the_bottom() {
        let mut rt = runtime(Transcript {
            answer: "A line of text.\n".repeat(25),
            empty: false,
        });
        rt.wheel(160.0, 80.0, 100.0);
        for _ in 0..60 {
            rt.frame(1.0 / 60.0);
        }
        let before = rt.presented("message-1-body").unwrap().0;
        rt.update(|state, tx| {
            state.answer.push_str("Another line.\n");
            tx.instant();
        });
        rt.frame(1.0 / 60.0);
        let after = rt.presented("message-1-body").unwrap().0;
        assert!(
            (after[1] - before[1]).abs() < 0.1,
            "reading position changed: {before:?} -> {after:?}"
        );
    }
    #[test]
    fn sentences_and_long_unbroken_words_wrap_within_the_viewport_and_rewrap_on_resize() {
        for answer in [
            "You have to remember who is running Syria, you don't know them so it's not like I'm going around asking a question. ".repeat(3),
            "longword".repeat(80),
        ] {
            let mut rt = runtime(Transcript { answer, empty: false });
            let narrow = rt.presented("message-1-body").unwrap().0;
            assert!(narrow[3] > 48.0, "long text must occupy multiple lines: {narrow:?}");
            assert!(narrow[0] >= 0.0 && narrow[0] + narrow[2] <= 320.1, "text escaped the viewport: {narrow:?}");
            rt.set_viewport((640.0, 160.0), 2.0);
            rt.frame(1.0 / 60.0);
            let wide = rt.presented("message-1-body").unwrap().0;
            assert!(wide[3] < narrow[3], "widening should reflow into fewer lines: {narrow:?} -> {wide:?}");
            assert!(wide[0] + wide[2] <= 640.1);
            rt.set_viewport((240.0, 160.0), 2.0);
            rt.frame(1.0 / 60.0);
            let tighter = rt.presented("message-1-body").unwrap().0;
            assert!(tighter[3] > wide[3], "narrowing should add lines");
            assert!(tighter[0] + tighter[2] <= 240.1);
        }
    }

    #[test]
    fn empty_state_contains_only_the_requested_headline() {
        let rt = runtime(Transcript {
            empty: true,
            ..Transcript::default()
        });
        let text: Vec<_> = rt.elements().into_iter().filter_map(|e| e.text).collect();
        assert_eq!(text, vec!["A little less intelligent."]);
    }
}
