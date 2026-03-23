import json

with open(r'e:\compare_mem0\mem0\evaluation\haiku\results\locomo_haiku\mem0_results.json', encoding='utf-8') as f:
    data = json.load(f)

items = [(q['question'], q['answer'], q.get('response','')) for v in data.values() for q in v]
print(f"Total: {len(items)}")
for question, answer, response in items[:30]:
    print(f"Q: {question[:70]}")
    print(f"A: {answer}")
    print(f"R: {response[:120]}")
    print()
