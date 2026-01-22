import unittest
import re

class TestStyles(unittest.TestCase):
    def test_container_max_width(self):
        with open("styles.css", "r") as f:
            content = f.read()
        
        # Regex to find .container { ... max-width: ... }
        match = re.search(r'\.container\s*\{[^}]*max-width:\s*([^;]+);', content, re.DOTALL)
        self.assertTrue(match, ".container class or max-width property not found")
        
        max_width = match.group(1).strip()
        self.assertEqual(max_width, "72rem", f"Expected max-width to be 72rem, but found {max_width}")

if __name__ == '__main__':
    unittest.main()
