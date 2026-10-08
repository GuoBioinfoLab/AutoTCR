# Compatibility imports

After installing the package, root-level `classify_model_1217.py` and `my_tokenizer.py` re-export the packaged classes for existing scripts. The class and parameter names remain compatible with complete state dictionaries.

The inference package does not execute the original model file's demonstration block or load arbitrary modules from private server paths. The original MultiClass.from_pretrained demonstration was incompatible with its nn.Module base and is replaced by the documented model factory.

