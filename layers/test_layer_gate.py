import unittest

from layer_gate import Gate


class testGate(unittest.TestCase):
    def test_combine_distributions(self):
        # Arrange 
        class_under_test = Gate()
        input_distribution_left = [0, 0.8, 0.2, 0, 0.7]
        input_distribution_right = [0, 0.2, 0.2, 0.6, 0.6]

        expected_output = [0, 0.523, 0.2, 0.277, 1.3]

        # Act 
        result = class_under_test.combine_distributions(input_distribution_left, input_distribution_right)

        # Assert
        self.assertEqual(result, expected_output)
        self.assertEqual(class_under_test.length_to_normalize, 2)


    def test_combine_distributions_list(self):
        # Arrange 
        class_under_test = Gate()
        input_distribution_1 = [0, 0.8, 0.2, 0, 0.7]
        input_distribution_2 = [0, 0.2, 0.2, 0.6, 0.6]

        input_distributions = [input_distribution_1, input_distribution_2, input_distribution_2]

        expected_output = [0, 0.421, 0.2, 0.379, 1.9]
        
        # Act 
        result = class_under_test.combine_distributions_list(input_distributions)

        # Assert
        self.assertEqual(result, expected_output)
        self.assertEqual(class_under_test.length_to_normalize, 3)

if __name__ == '__main__':
    unittest.main()
