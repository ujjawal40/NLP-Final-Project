from Generator import MedicalGenerator

generator = MedicalGenerator()

while True:
    query = input("Enter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # You can either pass an empty context or dummy context
    contexts = [""]  # You can customize this if needed
    answer = generator.generate(query, contexts)

    print(f"\nANSWER:\n{answer}\n")
