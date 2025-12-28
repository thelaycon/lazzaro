#!/usr/bin/env python3
"""
Test script to verify lazzaro installation and basic functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all new modular components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from lazzaro import Lazzaro, create_lazzaro, quick_chat
        print("✓ Main API imports successful")
    except ImportError as e:
        print(f"❌ Main API import failed: {e}")
        return False
    
    try:
        from lazzaro.core.config import MemoryConfig
        print("✓ Configuration imports successful")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    try:
        from lazzaro.core.resilience import CircuitBreaker, RetryManager, FallbackManager
        print("✓ Resilience components import successful")
    except ImportError as e:
        print(f"❌ Resilience import failed: {e}")
        return False
    
    try:
        from lazzaro.core.retriever import MemoryRetriever
        print("✓ Memory retriever import successful")
    except ImportError as e:
        print(f"❌ Memory retriever import failed: {e}")
        return False
    
    try:
        from lazzaro.core.consolidator import MemoryConsolidator
        print("✅ Memory consolidator import successful")
    except ImportError as e:
        print(f"❌ Memory consolidator import failed: {e}")
        return False
    
    try:
        from lazzaro.core.profile_manager import ProfileManager
        print("✅ Profile manager import successful")
    except ImportError as e:
        print(f"❌ Profile manager import failed: {e}")
        return False
    
    try:
        from lazzaro.core.orchestrator import MemoryOrchestrator
        print("✅ Memory orchestrator import successful")
    except ImportError as e:
        print(f"❌ Memory orchestrator import failed: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration system."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from lazzaro.core.config import MemoryConfig
        
        # Test default config
        config = MemoryConfig()
        assert config.enable_sharding == True
        assert config.max_buffer_size == 10
        assert config.db_dir == "db"
        print("✓ Default configuration works")
        
        # Test environment config
        os.environ["LAZZARO_MAX_BUFFER"] = "25"
        os.environ["LAZZARO_SHARDING"] = "false"
        
        env_config = MemoryConfig.from_env()
        assert env_config.max_buffer_size == 25
        assert env_config.enable_sharding == False
        print("✓ Environment configuration works")
        
        # Test serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "enable_sharding" in config_dict
        print("✓ Configuration serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_resilience():
    """Test resilience patterns."""
    print("\n🛡️  Testing resilience patterns...")
    
    try:
        from lazzaro.core.resilience import CircuitBreaker, RetryManager, FallbackManager
        
        # Test circuit breaker
        circuit = CircuitBreaker(failure_threshold=2, timeout=1)
        assert circuit.failure_threshold == 2
        print("✓ Circuit breaker creation works")
        
        # Test retry manager
        retry = RetryManager(max_retries=2, backoff_factor=0.1)
        assert retry.max_retries == 2
        print("✓ Retry manager creation works")
        
        # Test fallback manager
        fallback = FallbackManager()
        response = fallback.get_fallback_response("general")
        assert isinstance(response, str)
        assert len(response) > 0
        print("✓ Fallback manager works")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience test failed: {e}")
        return False


def test_api_structure():
    """Test API structure."""
    print("\n🚀 Testing API structure...")
    
    try:
        from lazzaro import Lazzaro, create_lazzaro, quick_chat
        
        # Test that classes exist and have expected methods
        assert callable(Lazzaro)
        assert callable(create_lazzaro)
        assert callable(quick_chat)
        print("✓ API functions exist")
        
        # Check if we can inspect the Lazzaro class (without instantiating)
        import inspect
        lazzaro_sig = inspect.signature(Lazzaro.__init__)
        expected_params = ['self', 'config', 'openai_api_key', 'llm_provider', 'embedding_provider', 'store', 'user_id']
        actual_params = list(lazzaro_sig.parameters.keys())
        
        for param in expected_params:
            if param in actual_params:
                print(f"✓ Parameter {param} exists")
            else:
                print(f"❌ Missing parameter: {param}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False


def test_data_structures():
    """Test data structures."""
    print("\n📊 Testing data structures...")
    
    try:
        from lazzaro.models.graph import Node, Edge
        
        # Test Node creation
        node = Node(id="test", content="test content")
        assert node.id == "test"
        assert node.content == "test content"
        assert node.type == "semantic"
        print("✓ Node creation works")
        
        # Test Edge creation
        edge = Edge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.weight == 1.0
        print("✓ Edge creation works")
        
        # Test serialization
        node_dict = node.to_dict()
        assert isinstance(node_dict, dict)
        assert node_dict['id'] == "test"
        print("✓ Node serialization works")
        
        edge_dict = edge.to_dict()
        assert isinstance(edge_dict, dict)
        assert edge_dict['source'] == "a"
        print("✓ Edge serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎉 Lazzaro Installation & Architecture Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_resilience,
        test_api_structure,
        test_data_structures
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎊 All tests passed! Installation successful!")
        print("\n📋 Next Steps:")
        print("  1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("  2. Try the quick example: python3 -c \"from lazzaro import quick_chat; print(quick_chat('your-key', 'Hello Lazzaro!'))\"")
        print("  3. Launch the dashboard: python3 -m lazzaro.dashboard.api")
        print("  4. Read the documentation: cat README.md")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())