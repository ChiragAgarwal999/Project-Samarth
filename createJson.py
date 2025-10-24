import json

# ---- Load your JSON file ----
with open("database.json", "r") as f:
    data = json.load(f)

# ---- Extract fields (column labels) ----
fields = [f["label"] for f in data["fields"]]
rows = data["data"]

sentences = []

# ---- Convert each row to a readable sentence ----
for row in rows:
    record = dict(zip(fields, row))

    subdivision = record.get("SUBDIVISION", "Unknown region")
    year = record.get("YEAR", "Unknown year")
    annual = record.get("ANNUAL", "N/A")

    # Prepare monthly details as text
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    monthly_info = ", ".join(
        [f"{m.title()}: {record.get(m, 'N/A')} mm" for m in months]
    )

    # Create descriptive sentence
    sentence = (
        f"In {year}, the subdivision '{subdivision}' received an annual rainfall "
        f"of {annual} mm. Monthly distribution was as follows: {monthly_info}."
    )

    sentences.append(sentence)

# ---- Save the sentences to a new JSON or text file ----
with open("rainfall_sentences.json", "w") as f:
    json.dump(sentences, f, indent=2)

print(f"✅ Successfully generated {len(sentences)} sentences!")
