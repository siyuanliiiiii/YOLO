from openai import OpenAI

client = OpenAI(api_key="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")

def ask(prompt: str, model: str = "gpt-4o") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(ask("You're supposed to see 8 screws. Four of them reside at the four corners and four resides at the middle on each side. Please mark anything you did not recognized in red."))