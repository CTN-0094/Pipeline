import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
from your_module import setup_logging, run_pipeline, scrape_log_to_csv

class TestPipeline(unittest.TestCase):

    @patch('your_module.logging.FileHandler')
    @patch('your_module.logging.StreamHandler')
    @patch('your_module.os.makedirs')
    def test_setup_logging(self, mock_makedirs, mock_stream_handler, mock_file_handler):
        """Test logging setup with a specific seed."""
        seed = 42
        log_filepath = setup_logging(seed)
        
        mock_makedirs.assert_called_once_with("logs", exist_ok=True)
        self.assertIn("pipeline_", log_filepath)
        self.assertIn("_42.log", log_filepath)
    
    @patch('your_module.load_datasets')
    @patch('your_module.preprocess_merged_data')
    @patch('your_module.create_and_merge_demographic_subsets')
    @patch('your_module.train_and_evaluate_models')
    @patch('your_module.log_pipeline_completion')
    @patch('your_module.setup_logging')
    def test_run_pipeline(self, mock_setup_logging, mock_log_completion, mock_train, mock_create_merge, mock_preprocess, mock_load):
        """Test the entire pipeline run with mocks."""
        seed = 42
        selected_outcome = "outcome_name"
        mock_setup_logging.return_value = "mock_log_path.log"
        
        run_pipeline(seed, selected_outcome)
        
        mock_setup_logging.assert_called_once_with(seed)
        mock_load.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_create_merge.assert_called_once()
        mock_train.assert_called_once()
        mock_log_completion.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="Global Seed set to: 42\nOutcome Name: TestOutcome\n")
    @patch('your_module.os.makedirs')
    @patch('your_module.parse_log_line', return_value=['0.85', '0.6', '76', '47', '51', '76', '0.61', '0.67'])
    def test_scrape_log_to_csv(self, mock_parse, mock_makedirs, mock_open):
        """Test log scraping and CSV generation."""
        log_filepaths = ["mock_log_path.log"]
        mock_makedirs.return_value = None
        
        scrape_log_to_csv(log_filepaths)
        
        # Check that the CSV file is created
        csv_file_path = f"log_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        mock_open.assert_called_with(os.path.join('logOutput', csv_file_path), 'w', newline='')

        # Check that parse_log_line is called correctly
        self.assertTrue(mock_parse.called)

if __name__ == '__main__':
    unittest.main()
