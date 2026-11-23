"""Token budget: named slots, newest turns kept, system prompt kept."""

import unittest

from app.memory.context import MESSAGE_OVERHEAD, build_prompt, estimate_tokens, message_tokens


def _prompt(history, *, system="Be brief.", context_size=400, reserved=32, notes="", retrieved=""):
    return build_prompt(
        history,
        system_prompt=system,
        context_size=context_size,
        reserved_output_tokens=reserved,
        pinned_notes=notes,
        retrieved_context=retrieved,
    )


class EstimateTests(unittest.TestCase):
    def test_rounds_characters_up(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("a"), 1)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 2)
        self.assertEqual(message_tokens("abcd"), 1 + MESSAGE_OVERHEAD)
        self.assertEqual(message_tokens(""), 0)


class BuildPromptTests(unittest.TestCase):
    def test_keeps_a_short_transcript(self):
        built = _prompt(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Ok"},
            ]
        )
        self.assertEqual(
            [item["content"] for item in built.messages],
            ["Be brief.", "Hi", "Ok"],
        )
        self.assertFalse(built.trimmed)
        self.assertEqual(built.tokens, sum(message_tokens(item["content"]) for item in built.messages))

    def test_ignores_system_and_blank_transcript_rows(self):
        built = _prompt(
            [
                {"role": "system", "content": "Ignore me."},
                {"role": "user", "content": "  Hi  "},
                {"role": "assistant", "content": "   "},
                {"role": "tool", "content": "nope"},
            ]
        )
        self.assertEqual([item["content"] for item in built.messages], ["Be brief.", "Hi"])

    def test_rejects_an_empty_transcript(self):
        with self.assertRaises(ValueError):
            _prompt([])

    def test_rejects_a_budget_with_no_room_for_a_reply(self):
        with self.assertRaises(ValueError):
            _prompt([{"role": "user", "content": "Hi"}], context_size=32, reserved=32)

    def test_drops_oldest_turns_first(self):
        old = "OLDQ"
        middle = "M" * 80
        newest = "NEWQ"
        system = "Go"
        budget = message_tokens(system) + message_tokens(old) + message_tokens(newest)
        built = build_prompt(
            [
                {"role": "user", "content": old},
                {"role": "user", "content": middle},
                {"role": "user", "content": newest},
            ],
            system_prompt=system,
            context_size=budget + 8,
            reserved_output_tokens=8,
        )
        self.assertEqual([item["content"] for item in built.messages], [system, newest])
        self.assertTrue(built.trimmed)

    def test_does_not_skip_a_turn_to_keep_an_older_one(self):
        """A gap in the middle is dropped, and the older turn stays dropped with it."""
        built = _prompt(
            [
                {"role": "user", "content": "OLDQ"},
                {"role": "assistant", "content": "then"},
                {"role": "user", "content": "M" * 400},
                {"role": "user", "content": "NEWQ"},
            ],
            context_size=message_tokens("Be brief.") + message_tokens("OLDQ") + message_tokens("then") + message_tokens("NEWQ") + 4,
            reserved=4,
        )
        self.assertEqual(built.messages[-1]["content"], "NEWQ")
        self.assertNotIn("OLDQ", [item["content"] for item in built.messages])
        self.assertTrue(built.trimmed)

    def test_keeps_the_tail_of_an_oversized_turn(self):
        text = "START" + ("x" * 100) + "END"
        system = "S"
        room = 3
        built = build_prompt(
            [{"role": "user", "content": text}],
            system_prompt=system,
            context_size=message_tokens(system) + MESSAGE_OVERHEAD + room + 4,
            reserved_output_tokens=4,
        )
        user = built.messages[-1]["content"]
        self.assertTrue(user.endswith("END"))
        self.assertNotIn("START", user)
        self.assertLessEqual(estimate_tokens(user), room)
        self.assertTrue(built.trimmed)
        self.assertEqual(built.messages[0]["content"], system)

    def test_never_drops_the_system_prompt(self):
        system = "S" * 80
        built = build_prompt(
            [{"role": "user", "content": "Hi"}],
            system_prompt=system,
            context_size=message_tokens(system),
            reserved_output_tokens=1,
        )
        self.assertEqual(built.messages, [{"role": "system", "content": system}])
        self.assertTrue(built.trimmed)
        self.assertGreater(built.tokens, message_tokens(system) - 1)

    def test_pinned_notes_and_retrieved_context_take_budget_first(self):
        notes = "PINNED"
        docs = "DOCS"
        newest = "NOW"
        older = "OLD" * 40
        fixed = message_tokens("Be brief.") + message_tokens(notes) + message_tokens(docs)
        built = build_prompt(
            [
                {"role": "user", "content": older},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": newest},
            ],
            system_prompt="Be brief.",
            context_size=fixed + message_tokens(newest) + 8,
            reserved_output_tokens=8,
            pinned_notes=notes,
            retrieved_context=docs,
        )
        self.assertEqual(
            [item["content"] for item in built.messages],
            ["Be brief.", notes, docs, newest],
        )
        self.assertEqual(
            [item["role"] for item in built.messages],
            ["system", "system", "user", "user"],
        )
        self.assertTrue(built.trimmed)

    def test_file_text_stays_out_of_the_system_prompt(self):
        built = build_prompt(
            [{"role": "user", "content": "Hi"}],
            system_prompt="Be brief.",
            context_size=400,
            reserved_output_tokens=32,
            retrieved_context="SECRET FILE TEXT",
            context_instruction="Reference only.",
        )
        system_text = "\n".join(item["content"] for item in built.messages if item["role"] == "system")
        self.assertIn("Reference only.", system_text)
        self.assertNotIn("SECRET FILE TEXT", system_text)
        self.assertEqual(built.messages[-2]["content"], "SECRET FILE TEXT")
        self.assertEqual(built.messages[-2]["role"], "user")

    def test_empty_optional_slots_add_no_messages(self):
        built = _prompt([{"role": "user", "content": "Hi"}], notes="  ", retrieved="")
        self.assertEqual([item["role"] for item in built.messages], ["system", "user"])

    def test_drops_old_turns_in_chunks_so_the_prefix_stays(self):
        """Six equal turns fit. The seventh drops four, and that start stays put."""
        system = "S"
        names = [f"{index:04d}" for index in range(11)]
        room = message_tokens(system) + 6 * message_tokens(names[0])

        def built(count):
            return build_prompt(
                [{"role": "user", "content": name} for name in names[:count]],
                system_prompt=system,
                context_size=room + 8,
                reserved_output_tokens=8,
            )

        self.assertEqual(
            [item["content"] for item in built(6).messages],
            [system, *names[:6]],
        )
        self.assertFalse(built(6).trimmed)
        self.assertEqual([item["content"] for item in built(7).messages], [system, *names[4:7]])
        self.assertEqual([item["content"] for item in built(8).messages][1], names[4])
        self.assertEqual([item["content"] for item in built(10).messages], [system, *names[4:10]])
        self.assertEqual([item["content"] for item in built(11).messages], [system, *names[8:11]])
