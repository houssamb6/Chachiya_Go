from agent.conversation import TunisiaTourismAgent
from config.settings import API_KEY
from config.content import WELCOMING


def main():
    print(WELCOMING)
    print()
    input("[ Press Enter to start your journey ] ")
    print()

    agent = TunisiaTourismAgent(api_key=API_KEY)

    reply, _ = agent.start()
    print(f"Yasmine: {reply}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print(f"\n[Reset]\n\n{WELCOMING}\n")
            input("[ Press Enter to start your journey ] ")
            print()
            reply, _ = agent.start()
            print(f"Yasmine: {reply}\n")
            continue

        reply, partners = agent.chat(user_input)
        print(f"\nYasmine: {reply}\n")

        if partners:
            print(partners)


if __name__ == "__main__":
    main()