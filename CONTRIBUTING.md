# Contributing

1. Create a focused branch for each change.
2. Update meaningful tests for changes to inference behavior.
3. Run python -m unittest discover -s tests -v and ruff check .
4. Check the built wheel when changing package assets.
5. Keep raw patient data, private paths, weight files and generated outputs out of Git.

Changes to tokenizer behavior, class order, fixed length, pooling or model assets require regression checks. Official weight updates should pin a new matched Hub revision and checksum.

