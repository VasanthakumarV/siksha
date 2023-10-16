release := env_var_or_default("RELEASE", "1")

CARGO_CMD := if release == "1" { "cargo r -r" } else { "cargo r" }

_run BIN *ARGS:
	{{CARGO_CMD}} --example {{BIN}} -- {{ARGS}}

data  *ARGS: (_run "data")
train *ARGS: (_run "model" "train" ARGS)
pred  *ARGS: (_run "model" "pred"  ARGS)

_viz TRUNK_CMD *ARGS:
	cd viz && {{TRUNK_CMD}} {{ if release == "1" { "--release" } else { "" } }} {{ARGS}}

serve *ARGS: (_viz "trunk serve" ARGS)
build *ARGS: (_viz "trunk build" ARGS)
