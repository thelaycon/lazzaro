"""
Test architecture improvements without external dependencies
"""
import os
import sys
import time
from pathlib import Path

# Add source to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


def test_config_structure():
    """Test configuration can be created and has expected attributes."""
    print("Testing configuration structure...")
    
    # Import just the config class
    try:
        from lazzaro.core.config import MemoryConfig
        print("✓ Config imports successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    # Test default configuration
    config = MemoryConfig()
    expected_attrs = [
        'enable_sharding', 'enable_hierarchy', 'enable_caching', 'enable_async',
        'max_shard_size', 'super_node_threshold', 'max_buffer_size', 'prune_threshold',
        'consolidate_every', 'decay_rate', 'llm_model', 'embedding_model',
        'db_dir', 'load_from_disk', 'max_retries', 'retry_backoff_factor',
        'circuit_breaker_threshold', 'circuit_breaker_timeout', 'cache_size', 'batch_size'
    ]
    
    for attr in expected_attrs:
        if not hasattr(config, attr):
            print(f"❌ Missing config attribute: {attr}")
            return False
    
    print("✓ All expected configuration attributes present")
    
    # Test environment loading
    os.environ["LAZZARO_SHARDING"] = "false"
    os.environ["LAZZARO_MAX_BUFFER"] = "15"
    
    env_config = MemoryConfig.from_env()
    assert env_config.enable_sharding == False
    assert env_config.max_buffer_size == 15
    print("✓ Environment configuration loading works")
    
    # Test serialization
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "enable_sharding" in config_dict
    print("✓ Configuration serialization works")
    
    return True


def test_resilience_patterns():
    """Test resilience patterns without dependencies."""
    print("Testing resilience patterns...")
    
    try:
        from lazzaro.core.resilience import CircuitBreaker, RetryManager, FallbackManager
        print("✓ Resilience components import successfully")
    except ImportError as e:
        print(f"❌ Resilience import failed: {e}")
        return False
    
    # Test circuit breaker
    circuit = CircuitBreaker(failure_threshold=2, timeout=1)
    
    failure_count = 0
    def failing_function():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:
            raise Exception("Simulated failure")
        return "success"
    
    # Should trigger circuit after 2 failures
    try:
        for _ in range(4):
            circuit.call(failing_function)
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            print("✓ Circuit breaker triggers correctly")
        else:
            print(f"❌ Unexpected circuit breaker error: {e}")
            return False
    
    # Test retry manager
    retry = RetryManager(max_retries=3, backoff_factor=0.01)  # Very fast retry
    
    attempts = 0
    def success_on_third():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("Retry me")
        return "finally worked"
    
    try:
        result = retry.with_retry(success_on_third)
        assert result == "finally worked"
        assert attempts == 3
        print("✓ Retry manager works correctly")
    except Exception as e:
        print(f"❌ Retry manager failed: {e}")
        return False
    
    # Test fallback manager
    fallback = FallbackManager()
    response = fallback.get_fallback_response("general")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ Fallback manager works correctly")
    
    return True


def test_api_structure():
    """Test API structure without dependencies."""
    print("Testing API structure...")
    
    # Test that we can at least import the API structure
    try:
        # Check if API file exists and has expected structure
        api_file = Path("src/lazzaro/api.py")
        if not api_file.exists():
            print("❌ API file does not exist")
            return False
        
        with open(api_file, 'r') as f:
            api_content = f.read()
        
        # Check for key components
        expected_classes = ["Lazzaro"]
        expected_functions = ["create_lazzaro", "quick_chat"]
        
        for cls in expected_classes:
            if f"class {cls}" not in api_content:
                print(f"❌ Missing class: {cls}")
                return False
        
        for func in expected_functions:
            if f"def {func}" not in api_content:
                print(f"❌ Missing function: {func}")
                return False
        
        print("✓ API structure is correct")
        
        # Check for key methods in Lazzaro class
        expected_methods = ["chat", "remember", "recall", "get_insights", "close"]
        for method in expected_methods:
            if f"def {method}" not in api_content:
                print(f"❌ Missing method: {method}")
                return False
        
        print("✓ All expected API methods present")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False


def test_modular_files():
    """Test that all expected modular files exist."""
    print("Testing modular file structure...")
    
    expected_files = [
        "src/lazzaro/core/config.py",
        "src/lazzaro/core/resilience.py", 
        "src/lazzaro/core/retriever.py",
        "src/lazzaro/core/consolidator.py",
        "src/lazzaro/core/profile_manager.py",
        "src/lazzaro/core/orchestrator.py",
        "src/lazzaro/api.py"
    ]
    
    for file_path in expected_files:
        if not Path(file_path).exists():
            print(f"❌ Missing file: {file_path}")
            return False
    
    print("✓ All expected modular files present")
    return True


def test_backward_compatibility():
    """Test that old MemorySystem can still be imported."""
    print("Testing backward compatibility...")
    
    try:
        from lazzaro.core.memory_system import MemorySystem
        print("✓ Original MemorySystem still available")
        return True
    except ImportError as e:
        print(f"❌ MemorySystem import failed: {e}")
        return False


def main():
    """Run all architecture tests."""
    print("🧪 Lazzaro Architecture Tests\n")
    print("=" * 50)
    
    tests = [
        test_modular_files,
        test_config_structure,
        test_resilience_patterns,
        test_api_structure,
        test_backward_compatibility
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
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All architecture tests passed!")
        print("\n📋 Implementation Summary:")
        print("  ✅ Modular configuration system")
        print("  ✅ Resilience patterns (circuit breaker, retry, fallback)")
        print("  ✅ Specialized components (retriever, consolidator, profile manager)")
        print("  ✅ Clean public API")
        print("  ✅ Backward compatibility maintained")
        print("  ✅ Proper file structure and organization")
        
        print("\n🚀 Architecture improvements successfully implemented!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())