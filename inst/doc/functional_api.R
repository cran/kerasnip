## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = reticulate::py_module_available("keras")
)

## ----setup--------------------------------------------------------------------
library(kerasnip)
library(tidymodels)
library(keras3)

# Silence the startup messages from remove_keras_spec
options(kerasnip.show_removal_messages = FALSE)

## ----define-blocks-functional-two-input---------------------------------------
# Input blocks for two distinct inputs
input_block_1 <- function(input_shape) {
  layer_input(shape = input_shape, name = "input_1")
}

input_block_2 <- function(input_shape) {
  layer_input(shape = input_shape, name = "input_2")
}

# Dense paths for each input
dense_path_1 <- function(tensor, units = 16) {
  tensor |> layer_dense(units = units, activation = "relu")
}

dense_path_2 <- function(tensor, units = 16) {
  tensor |> layer_dense(units = units, activation = "relu")
}

# A block to join two tensors
concat_block <- function(input_a, input_b) {
  layer_concatenate(list(input_a, input_b))
}

# The final output block for regression
output_block_1 <- function(tensor) {
  layer_dense(tensor, units = 1, name = "output_1")
}

output_block_2 <- function(tensor) {
  layer_dense(tensor, units = 1, name = "output_2")
}

## ----create-spec-functional-two-input-----------------------------------------
model_name <- "two_output_reg_spec" # Changed model name
# Clean up the spec when the vignette is done knitting
on.exit(remove_keras_spec(model_name), add = TRUE)

create_keras_functional_spec(
  model_name = model_name,
  layer_blocks = list(
    input_1 = input_block_1,
    input_2 = input_block_2,
    processed_1 = inp_spec(dense_path_1, "input_1"),
    processed_2 = inp_spec(dense_path_2, "input_2"),
    concatenated = inp_spec(
      concat_block,
      c(input_a = "processed_1", input_b = "processed_2")
    ),
    output_1 = inp_spec(output_block_1, "concatenated"), # New output block 1
    output_2 = inp_spec(output_block_2, "concatenated") # New output block 2
  ),
  mode = "regression" # Still regression, but will have two columns in y
)

## ----fit-functional-two-input-------------------------------------------------
# We can override the default `units` for each path.
spec <- two_output_reg_spec( # Changed spec name
  processed_1_units = 16,
  processed_2_units = 8,
  fit_epochs = 10,
  fit_verbose = 0 # Suppress fitting output in vignette
) |>
  set_engine("keras")

print(spec)

# Prepare dummy data with two inputs and two outputs
set.seed(123)
x_data_1 <- matrix(runif(100 * 5), ncol = 5)
x_data_2 <- matrix(runif(100 * 3), ncol = 3)
y_data_1 <- runif(100)
y_data_2 <- runif(100) # New second output

# For tidymodels, inputs and outputs need to be in a data frame,
# potentially as lists of matrices
train_df <- tibble::tibble(
  input_1 = lapply(
    seq_len(nrow(x_data_1)),
    function(i) x_data_1[i, , drop = FALSE]
  ),
  input_2 = lapply(
    seq_len(nrow(x_data_2)),
    function(i) x_data_2[i, , drop = FALSE]
  ),
  output_1 = y_data_1, # Named output 1
  output_2 = y_data_2 # Named output 2
)

rec <- recipe(output_1 + output_2 ~ input_1 + input_2, data = train_df)
wf <- workflow() |>
  add_recipe(rec) |>
  add_model(spec)

fit_obj <- fit(wf, data = train_df)

# Predict on new data
new_data_df <- tibble::tibble(
  input_1 = lapply(seq_len(5), function(i) matrix(runif(5), ncol = 5)),
  input_2 = lapply(seq_len(5), function(i) matrix(runif(3), ncol = 3))
)
predict(fit_obj, new_data = new_data_df)

## ----compile-grid-debug-functional--------------------------------------------
# Create a spec instance
spec <- two_output_reg_spec( # Changed spec name
  processed_1_units = 16,
  processed_2_units = 8
)

# Prepare dummy data with two inputs and two outputs
x_dummy_1 <- matrix(runif(10 * 5), ncol = 5)
x_dummy_2 <- matrix(runif(10 * 3), ncol = 3)
y_dummy_1 <- runif(10)
y_dummy_2 <- runif(10) # New second output

# For tidymodels, inputs and outputs need to be in a data frame,
# potentially as lists of matrices
x_dummy_df <- tibble::tibble(
  input_1 = lapply(
    seq_len(nrow(x_dummy_1)),
    function(i) x_dummy_1[i, , drop = FALSE]
  ),
  input_2 = lapply(
    seq_len(nrow(x_dummy_2)),
    function(i) x_dummy_2[i, , drop = FALSE]
  )
)
y_dummy_df <- tibble::tibble(output_1 = y_dummy_1, output_2 = y_dummy_2)

# Use compile_keras_grid to get the model
compilation_results <- compile_keras_grid(
  spec = spec,
  grid = tibble::tibble(),
  x = x_dummy_df,
  y = y_dummy_df
)

# Print the summary
compilation_results |>
  select(compiled_model) |>
  pull() |>
  pluck(1) |>
  summary()

## ----grid-debug-plot, eval=FALSE----------------------------------------------
# compilation_results |>
#   select(compiled_model) |>
#   pull() |>
#   pluck(1) |>
#   plot(show_shapes = TRUE)

