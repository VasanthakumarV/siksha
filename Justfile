nn +ARGS='train':
	cargo r -r -p nn --bin nn -- {{ARGS}}

test:
	cargo r -r -p nn --bin test
