import unittest
from services.memory import ConversationMemory

class TestConversationMemory(unittest.TestCase):
    
    def test_window_cut_limit(self):
        # max_turns=2 means max_messages = 4 (2 user, 2 model turns)
        memory = ConversationMemory(max_turns=2)
        sid = "test_window"
        
        memory.add_turn(sid, "Q1", "A1")
        memory.add_turn(sid, "Q2", "A2")
        memory.add_turn(sid, "Q3", "A3")
        
        history = memory.get_history(sid)
        # Should only retain last 2 turns (Q2/A2 and Q3/A3) -> 4 messages total
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "Q2")
        self.assertEqual(history[2]["content"], "Q3")

    def test_new_or_missing_session(self):
        memory = ConversationMemory()
        # Missing / None session_id should return empty list and do nothing
        history = memory.get_history(None)
        self.assertEqual(history, [])
        
        memory.add_turn(None, "Q1", "A1")
        self.assertEqual(len(memory.sessions), 0)

    def test_two_sessions_do_not_mix(self):
        memory = ConversationMemory()
        sid_a = "session_alpha"
        sid_b = "session_beta"
        
        memory.add_turn(sid_a, "Question A", "Answer A")
        memory.add_turn(sid_b, "Question B", "Answer B")
        
        history_a = memory.get_history(sid_a)
        history_b = memory.get_history(sid_b)
        
        self.assertEqual(len(history_a), 2)
        self.assertEqual(history_a[0]["content"], "Question A")
        
        self.assertEqual(len(history_b), 2)
        self.assertEqual(history_b[0]["content"], "Question B")

if __name__ == "__main__":
    unittest.main()