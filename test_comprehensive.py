"""
Comprehensive test runner for lazzaro without external dependencies
"""
import sys
import os
from pathlib import Path

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_file_structure():
    """Test that all expected files exist."""
    print("📁 Testing file structure...")
    
    expected_files = [
        "src/lazzaro/__init__.py",
        "src/lazzaro/api.py", 
        "src/lazzaro/core/__init__.py",
        "src/lazzaro/core/config.py",
        "src/lazzaro/core/resilience.py",
        "src/lazzaro/core/retriever.py",
        "src/lazzaro/core/consolidator.py",
        "src/lazzaro/core/profile_manager.py",
        "src/lazzaro/core/orchestrator.py",
        "src/lazzaro/core/resilient_providers.py",
        "src/lazzaro/core/memory_shard.py",
        "src/lazzaro/core/buffer_graph.py",
        "src/lazzaro/core/profile.py",
        "src/lazzaro/core/query_cache.py",
        "src/lazzaro/core/vector_store.py",
        "src/lazzaro/core/interfaces.py",
        "src/lazzaro/models/graph.py",
        "src/lazzaro/cli/main.py",
        "src/lazzaro/dashboard/api.py",
        "src/lazzaro/dashboard/templates/index.html",
        "src/lazzaro/integrations/langchain_integration.py",
        "src/lazzaro/integrations/langgraph_integration.py",
        "tests/"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = Path(file_path)
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✓ All expected files present")
        return True


def test_imports_basic():
    """Test basic imports that don't require external deps."""
    print("\n🔍 Testing basic imports...")
    
    try:
        # Test data structures (no external deps)
        from lazzaro.models.graph import Node, Edge
        print("✓ Data structures imported successfully")
        
        # Test interfaces
        from lazzaro.core.interfaces import LLMProvider, EmbeddingProvider, Store
        print("✓ Interfaces imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False


def test_file_syntax():
    """Test that Python files have valid syntax."""
    print("\n🔤 Testing file syntax...")
    
    python_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}:{e.lineno} - {e.msg}")
        except Exception as e:
            # Other compilation errors are fine for now
            pass
    
    if syntax_errors:
        print(f"❌ Syntax errors: {syntax_errors}")
        return False
    else:
        print(f"✓ All {len(python_files)} Python files have valid syntax")
        return True


def test_configuration():
    """Test configuration system."""
    print("\n⚙️  Testing configuration system...")
    
    try:
        from lazzaro.core.config import MemoryConfig
        
        # Test default config
        config = MemoryConfig()
        required_attrs = [
            'enable_sharding', 'enable_hierarchy', 'enable_caching', 'enable_async',
            'max_shard_size', 'super_node_threshold', 'max_buffer_size', 'prune_threshold',
            'consolidate_every', 'decay_rate', 'llm_model', 'embedding_model',
            'db_dir', 'load_from_disk', 'max_retries', 'retry_backoff_factor',
            'circuit_breaker_threshold', 'circuit_breaker_timeout', 'cache_size', 'batch_size'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"❌ Missing config attribute: {attr}")
                return False
        
        print("✓ Configuration has all required attributes")
        
        # Test methods
        config_dict = config.to_dict()
        if not isinstance(config_dict, dict):
            print("❌ to_dict() method failed")
            return False
        
        print("✓ Configuration methods work")
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False


def test_resilience_patterns():
    """Test resilience patterns without external deps."""
    print("\n🛡️  Testing resilience patterns...")
    
    try:
        from lazzaro.core.resilience import CircuitBreaker, RetryManager, FallbackManager
        
        # Test circuit breaker
        circuit = CircuitBreaker(failure_threshold=3, timeout=60)
        if circuit.failure_threshold != 3 or circuit.timeout != 60:
            print("❌ Circuit breaker initialization failed")
            return False
        
        # Test retry manager
        retry = RetryManager(max_retries=5, backoff_factor=2.0)
        if retry.max_retries != 5 or retry.backoff_factor != 2.0:
            print("❌ Retry manager initialization failed")
            return False
        
        # Test fallback manager
        fallback = FallbackManager()
        response = fallback.get_fallback_response("general")
        if not isinstance(response, str) or len(response) == 0:
            print("❌ Fallback manager failed")
            return False
        
        print("✓ Resilience patterns work correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Resilience import failed: {e}")
        return False


def test_api_structure():
    """Test API structure."""
    print("\n🚀 Testing API structure...")
    
    try:
        from lazzaro.api import Lazzaro, create_lazzaro, quick_chat
        
        # Check that functions are callable
        if not callable(Lazzaro):
            print("❌ Lazzaro class not callable")
            return False
        
        if not callable(create_lazzaro):
            print("❌ create_lazzaro function not callable")
            return False
        
        if not callable(quick_chat):
            print("❌ quick_chat function not callable")
            return False
        
        print("✓ API functions are callable")
        return True
        
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False


def test_readme():
    """Test README structure."""
    print("\n📖 Testing README...")
    
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("❌ README.md not found")
        return False
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    required_sections = [
        "Quick Start",
        "Installation", 
        "Architecture",
        "Usage Examples",
        "API Reference",
        "Production Deployment"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ README missing sections: {missing_sections}")
        return False
    else:
        print("✓ README contains all required sections")
        return True


def test_pyproject():
    """Test pyproject.toml structure."""
    print("\n📦 Testing pyproject.toml...")
    
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    with open(pyproject_path, 'r') as f:
        pyproject_content = f.read()
    
    required_sections = [
        "[build-system]",
        "[project]", 
        "dependencies =",
        "lazzaro-cli",
        "lazzaro-dashboard"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in pyproject_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ pyproject.toml missing sections: {missing_sections}")
        return False
    else:
        print("✓ pyproject.toml has all required sections")
        return True


def main():
    """Run comprehensive tests."""
    print("🎉 Lazzaro Comprehensive Test Suite")
    print("=" * 70)
    print("Testing installation, architecture, and documentation...")
    print("=" * 70)
    
    tests = [
        test_file_structure,
        test_imports_basic,
        test_file_syntax,
        test_configuration,
        test_resilience_patterns,
        test_api_structure,
        test_readme,
        test_pyproject
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
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎊 ALL TESTS PASSED!")
        print("\n🚀 Lazzaro is ready for use!")
        print("\n📋 Quick Start Guide:")
        print("  1. Install dependencies: pip install numpy openai lancedb networkx matplotlib plotly fastapi uvicorn jinja2 pyarrow pyyaml")
        print("  2. Set API key: export OPENAI_API_KEY='your-key'")
        print("  3. Test: python3 -c \"import lazzaro; print('✅ Installation successful!')\"")
        print("  4. Try demo: python3 demonstrate_architecture.py")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())