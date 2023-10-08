data *ARGS: (_run "nn" "data")

train *ARGS: (_run "nn" "model" "train" ARGS)

pred *ARGS: (_run "nn" "model" "pred" ARGS)

_run PKG BIN *ARGS:
	cargo r -r -p {{PKG}} --example {{BIN}} -- {{ARGS}}

viz *ARGS:
	cd viz && trunk serve --release {{ARGS}}
