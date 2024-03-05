import unittest
from helper import nonzero_equal, classifier 

class TestBugClassifier(unittest.TestCase):
    def test_nonzero_equal(self):
        summary = "keyboard shortcut debug start stop publish work server view"
        severe_words = ['keyboard', 'debug', 'stop', 'work']
        nonsevere_words = ['shortcut', 'start', 'publish', 'server']

        result = nonzero_equal(summary, severe_words, nonsevere_words)
        self.assertIn(result, ['Severe', 'NonSevere'])
        print(f"Bug severity: {summary} {result}")

    def test_classifier(self):
        Summary = "- keyboard shortcut debug start stop publish work server view"
        severedictionary_list = ['keyboard', 'debug', 'stop', 'work']
        nonseveredictionary_list = ['shortcut', 'start', 'publish', 'server']

        result = classifier(Summary, severedictionary_list, nonseveredictionary_list)
        self.assertIn(result, ['Severe', 'NonSevere'])
        print(f"Bug severity: {Summary} {result}")

if __name__ == "__main__":
    unittest.main()

# severedictionary_list = ["target", "entity", "link", "nonexistent"]
# nonseveredictionary_list = ["work", "properly", "not", "exist"]
# summary_example = "target entity link not work properly nonexistent package"

# # Test the classifier function
# def test_nonzero_equal(self):
#     result = classifier(summary_example, severedictionary_list, nonseveredictionary_list)
#     self.assertIn(result, ['Severe', 'NonSevere'])
#     print(f"Bug severity: {summary_example} {result}")