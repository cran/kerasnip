## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = reticulate::py_module_available("keras") &&
    requireNamespace("keras3", quietly = TRUE) &&
    requireNamespace("ggplot2", quietly = TRUE) &&
    requireNamespace("tune", quietly = TRUE) &&
    requireNamespace("dials", quietly = TRUE) &&
    requireNamespace("parsnip", quietly = TRUE) &&
    requireNamespace("workflows", quietly = TRUE) &&
    requireNamespace("recipes", quietly = TRUE) &&
    requireNamespace("rsample", quietly = TRUE)
)

## -----------------------------------------------------------------------------
#> Error in `dplyr::rename()`:
#> ! Names must be unique.
#> ✖ These names are duplicated:
#>   * "# Hidden Units" at locations 1 and 2.

## ----model_definition---------------------------------------------------------
library(kerasnip)
library(keras3)
library(parsnip)
library(dials)
library(workflows)
library(recipes)
library(rsample)
library(tune)
library(ggplot2)

# Define a spec with multiple hidden unit parameters
model_name <- "autoplot_unique_spec"
# Clean up the spec if it already exists from a previous run
if (exists(model_name, mode = "function")) {
  suppressMessages(remove_keras_spec(model_name))
}

input_block <- function(model, input_shape) {
  keras3::keras_model_sequential(input_shape = input_shape)
}

dense_block <- function(model, units = 10) {
  model |> keras3::layer_dense(units = units)
}

output_block <- function(model, num_classes) {
  model |>
    keras3::layer_dense(units = num_classes, activation = "softmax")
}

create_keras_sequential_spec(
  model_name = model_name,
  layer_blocks = list(
    input = input_block,
    dense1 = dense_block,
    dense2 = dense_block,
    output = output_block
  ),
  mode = "classification"
)

# Now, create the model specification and assign unique IDs for tuning
tune_spec <- autoplot_unique_spec(
  dense1_units = tune(id = "dense_layer_one_units"),
  dense2_units = tune(id = "dense_layer_two_units")
) |>
  set_engine("keras")

print(tune_spec)

## ----tuning_setup-------------------------------------------------------------
# Set up workflow and tuning grid
rec <- recipes::recipe(Species ~ ., data = iris)
tune_wf <- workflows::workflow(rec, tune_spec)

params <- tune::extract_parameter_set_dials(tune_wf)

# Update the parameter ranges using kerasnip::hidden_units
# The `id`s provided in tune() are automatically detected and used here.
params <- params |>
  update(
    dense_layer_one_units = hidden_units(range = c(4L, 8L)),
    dense_layer_two_units = hidden_units(range = c(4L, 8L))
  )

grid <- dials::grid_regular(params, levels = 2)
control <- tune::control_grid(save_pred = FALSE, verbose = FALSE)

print(grid)

## ----run_tuning---------------------------------------------------------------
# Run tuning
tune_res <- tune::tune_grid(
  tune_wf,
  resamples = rsample::vfold_cv(iris, v = 2),
  grid = grid,
  control = control
)

print(tune_res)

## ----autoplot_results, fig.width=7, fig.height=5------------------------------
# Assert that autoplot works without error
ggplot2::autoplot(tune_res)

## ----cleanup, include=FALSE---------------------------------------------------
suppressMessages(remove_keras_spec(model_name))

