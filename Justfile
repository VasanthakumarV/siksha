train *ARGS: (_run "nn" "nn" ARGS)

_run PKG BIN ARGS:
	cargo r -r -p {{PKG}} --bin {{BIN}} -- {{ARGS}}
