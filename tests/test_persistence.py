import unittest
import os
import sys
# Add src to path if not installed yet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import shutil
import pickle
from lazzaro.core.persistence import PersistenceManager
from lazzaro.core.memory_system import MemorySystem
from unittest.mock import MagicMock, patch

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_db"
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)
        self.pm = PersistenceManager(db_dir=self.test_db, filename="test.pkl")

    def tearDown(self):
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)

    def test_save_load(self):
        data = {"foo": "bar", "num": 123}
        self.assertTrue(self.pm.save(data))
        self.assertTrue(os.path.exists(os.path.join(self.test_db, "test.pkl")))
        
        loaded = self.pm.load()
        self.assertEqual(loaded, data)

    def test_backup_creation(self):
        data1 = {"v": 1}
        self.pm.save(data1)
        self.assertTrue(os.path.exists(os.path.join(self.test_db, "test.pkl")))
        self.assertFalse(os.path.exists(os.path.join(self.test_db, "test.pkl.bak")))

        data2 = {"v": 2}
        self.pm.save(data2)
        self.assertTrue(os.path.exists(os.path.join(self.test_db, "test.pkl.bak")))

class TestMemorySystemPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_ms_db"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('lazzaro.core.memory_system.PersistenceManager')
    def test_auto_load(self, MockPM):
        # Setup mock
        mock_instance = MockPM.return_value
        mock_instance.load.return_value = {
            "node_counter": 999
        }
        
        # Init memory system
        ms = MemorySystem(openai_api_key="fake", enable_async=False, load_from_disk=True)
        
        # Verify load was called
        mock_instance.load.assert_called_once()
        self.assertEqual(ms.node_counter, 999)

    @patch('lazzaro.core.memory_system.PersistenceManager')
    def test_auto_save(self, MockPM):
        # Setup mock
        mock_instance = MockPM.return_value
        print(f"DEBUG: MockPM called. instance={mock_instance}")
        
        ms = MemorySystem(openai_api_key="fake", enable_async=False, load_from_disk=False)
        print(f"DEBUG: ms.persistence type: {type(ms.persistence)}")
        ms.node_counter = 50
        
        # Trigger save
        ms._save_to_persistence()
        
        mock_instance.save.assert_called_once()
        args = mock_instance.save.call_args[0][0]
        self.assertEqual(args['node_counter'], 50)

if __name__ == '__main__':
    unittest.main()
