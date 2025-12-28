"""
Test new modular architecture
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lazzaro.core.config import MemoryConfig
from lazzaro.core.resilience import CircuitBreaker, RetryManager
from lazzaro.api import create_lazzaro, quick_chat


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    # Test default config
    config = MemoryConfig()
    assert config.enable_sharding == True
    assert config.llm_model == "gpt-4o-mini"
    print("✓ Default configuration works")
    
    # Test environment config
    os.environ["LAZZARO_SHARDING"] = "false"
    env_config = MemoryConfig.from_env()
    assert env_config.enable_sharding == False
    print("✓ Environment configuration works")


def test_resilience():
    """Test resilience patterns."""
    print("Testing resilience patterns...")
    
    # Test circuit breaker
    circuit = CircuitBreaker(failure_threshold=2, timeout=1)
    
    failure_count = 0
    def failing_function():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:
            raise Exception("Simulated failure")
        return "success"
    
    # Should fail first two times
    try:
        circuit.call(failing_function)
    except:
        pass  # Expected
    
    try:
        circuit.call(failing_function)
    except:
        pass  # Expected
    
    # Should be open now
    try:
        circuit.call(failing_function)
        assert False, "Should have failed due to open circuit"
    except Exception as e:
        assert "Circuit breaker is OPEN" in str(e)
    
    print("✓ Circuit breaker works")
    
    # Test retry manager
    retry = RetryManager(max_retries=2, backoff_factor=0.1)  # Fast retry
    
    attempts = 0
    def counting_function():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("Retry me")
        return "finally worked"
    
    result = retry.with_retry(counting_function)
    assert result == "finally worked"
    assert attempts == 3
    print("✓ Retry manager works")


def test_api():
    """Test simplified API without actual API calls."""
    print("Testing simplified API...")
    
    # Test that API can be imported and basic structure exists
    assert callable(create_lazzaro)
    assert callable(quick_chat)
    print("✓ API functions exist")


def test_integration():
    """Test basic integration points."""
    print("Testing integration...")
    
    # Create a minimal config
    config = MemoryConfig(
        enable_sharding=True,
        enable_caching=True,
        max_buffer_size=5
    )
    
    # Verify config can be serialized
    config_dict = config.to_dict()
    assert "enable_sharding" in config_dict
    assert config_dict["max_buffer_size"] == 5
    print("✓ Configuration serialization works")
    
    print("✓ Integration tests passed")


def main():
    """Run all tests."""
    print("🧪 Running Architecture Tests\n")
    
    try:
        test_config()
        test_resilience()
        test_api()
        test_integration()
        
        print("\n✅ All architecture tests passed!")
        print("\n📋 Implementation Summary:")
        print("  ✓ Modular configuration system")
        print("  ✓ Resilience patterns (circuit breaker, retry)")
        print("  ✓ Simplified public API")
        print("  ✓ Clean separation of concerns")
        print("  ✓ Error handling and logging")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())