import os
import sys
from ..core.memory_system import MemorySystem

def interactive_chat():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        # Try to read from input if not set, for convenience
        api_key = input("Enter OpenAI API Key: ").strip()
        if not api_key:
            print("API Key required. Exiting.")
            return

    print("=" * 60)
    print("  SCALABLE MEMORY SYSTEM - CLI")
    print("=" * 60)
    print("\nCommands: /start, /end, /stats, /profile, /memories, /consolidate")
    print("          /merge, /prune, /config, /set, /save, /load, /quit")

    memory = MemorySystem(
        api_key,
        enable_sharding=True,
        enable_hierarchy=True,
        enable_caching=True,
        enable_async=True,
        max_buffer_size=10,
        prune_threshold=0.5
    )


    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.lower().split()
                cmd = parts[0]

                if cmd == "/quit":
                    if memory.conversation_active:
                        print("\n" + memory.end_conversation())
                    print("\n👋 Goodbye!")
                    break

                elif cmd == "/start":
                    print("\n" + memory.start_conversation())

                elif cmd == "/end":
                    print("\n" + memory.end_conversation())

                elif cmd == "/stats":
                    print(memory.display_stats())

                elif cmd == "/profile":
                    print(memory.display_profile())

                elif cmd == "/memories":
                    limit = int(parts[1]) if len(parts) > 1 else 10
                    print(memory.display_memories(limit=limit))

                elif cmd == "/consolidate":
                    print("\n" + memory.run_consolidation())

                elif cmd == "/merge":
                    print("\n🔄 Merging similar nodes...")
                    merged = memory._merge_similar_nodes()
                    print(f"✓ Merged {merged} similar nodes")

                elif cmd == "/prune":
                    threshold = float(parts[1]) if len(parts) > 1 else memory.prune_threshold
                    print(f"\n🔄 Pruning edges below {threshold}...")
                    pruned = memory.buffer.prune_weak_edges(threshold=threshold)
                    print(f"✓ Pruned {pruned} weak edges")

                elif cmd == "/config":
                    print("\n⚙️ Configuration:")
                    for param in ['max_buffer_size', 'prune_threshold', 'consolidate_every',
                                  'auto_consolidate', 'auto_prune', 'enable_sharding',
                                  'enable_hierarchy', 'enable_caching', 'enable_async']:
                        print(f"  • {param}: {getattr(memory, param)}")

                elif cmd == "/set":
                    if len(parts) < 3:
                        print("⚠ Usage: /set <parameter> <value>")
                        continue
                    param, value_str = parts[1], parts[2]
                    try:
                        if hasattr(memory, param):
                            val_type = type(getattr(memory, param))
                            if val_type == bool:
                                value = value_str.lower() in ['true', '1', 'on', 'yes']
                            else:
                                value = val_type(value_str)
                            setattr(memory, param, value)
                            print(f"✓ Set {param} = {value}")
                        else:
                            print(f"⚠ Unknown parameter: {param}")
                    except ValueError:
                        print(f"⚠ Invalid value for {param}")

                elif cmd == "/save":
                    ms = memory
                    ms._save_to_persistence()
                    filename = parts[1] if len(parts) > 1 else "memory_state.json"
                    # Also save JSON for export if requested or default
                    print("\n" + memory.save_state(filename))
                    print(f"✓ Also saved to optimized persistence at {ms.persistence.filepath}")

                elif cmd == "/load":
                    filename = parts[1] if len(parts) > 1 else None
                    if filename:
                        print("\n" + memory.load_state(filename))
                    else:
                        memory._load_from_persistence()
                        print(f"\n✓ Reloaded from optimized persistence at {memory.persistence.filepath}")

                elif cmd == "/help":
                    print("Available commands: /start, /end, /stats, /profile, /memories [n], /consolidate, /merge, /prune [thresh], /config, /set <k> <v>, /save, /load")

            else:
                # Use streaming response for chat
                first_token = True
                print("Assistant: ", end="", flush=True)
                
                for event in memory.chat_stream(user_input):
                    if event["type"] == "token":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "info":
                        # Print info/debug messages before the assistant starts speaking, or on a new line
                        if first_token:
                            print(f"\n{event['content']}") 
                        else:
                            # If we are already printing tokens, maybe skip or print at end?
                            # For now, let's keep info quiet during generation or print clearly.
                            # The chat_stream yields info first, so this logic is slightly redundant but safe.
                            pass 
                print() # Newline at end


        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")

def entry_point():
    interactive_chat()

if __name__ == "__main__":
    interactive_chat()
