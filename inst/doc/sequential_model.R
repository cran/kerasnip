## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = reticulate::py_module_available("keras")
)
# Suppress verbose Keras output for the vignette
options(keras.fit_verbose = 0)
set.seed(123)

## ----load-packages------------------------------------------------------------
library(kerasnip)
library(tidymodels)
library(keras3)

## ----define-simple-blocks-----------------------------------------------------
# The first block must initialize the model. `input_shape`
# is passed automatically.
input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}

# A reusable block for hidden layers. `units` will become a tunable parameter.
hidden_block <- function(model, units = 32, activation = "relu") {
  model |> layer_dense(units = units, activation = activation)
}

# The output block. `num_classes` is passed automatically for classification.
output_block <- function(model, num_classes, activation = "softmax") {
  model |> layer_dense(units = num_classes, activation = activation)
}

## ----create-simple-spec-------------------------------------------------------
create_keras_sequential_spec(
  model_name = "my_simple_mlp",
  layer_blocks = list(
    input = input_block,
    hidden_1 = hidden_block,
    hidden_2 = hidden_block,
    output = output_block
  ),
  mode = "classification"
)

## ----compile-grid-debug-------------------------------------------------------
# Define CNN layer blocks
cnn_input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}
cnn_conv_block <- function(
  model,
  filters = 32,
  kernel_size = 3,
  activation = "relu"
) {
  model |>
    layer_conv_2d(
      filters = filters,
      kernel_size = kernel_size,
      activation = activation
    )
}
cnn_pool_block <- function(model, pool_size = 2) {
  model |> layer_max_pooling_2d(pool_size = pool_size)
}
cnn_flatten_block <- function(model) {
  model |> layer_flatten()
}
cnn_output_block <- function(model, num_classes, activation = "softmax") {
  model |> layer_dense(units = num_classes, activation = activation)
}

# Create the kerasnip spec function
create_keras_sequential_spec(
  model_name = "my_cnn",
  layer_blocks = list(
    input = cnn_input_block,
    conv1 = cnn_conv_block,
    pool1 = cnn_pool_block,
    flatten = cnn_flatten_block,
    output = cnn_output_block
  ),
  mode = "classification"
)

# Create a spec instance for a 28x28x1 image
cnn_spec <- my_cnn(
  conv1_filters = 32, conv1_kernel_size = 5,
  compile_loss = "categorical_crossentropy",
  compile_optimizer = "adam"
)

# Prepare dummy data with the correct shape.
# We create a list of 28x28x1 arrays.
x_dummy_list <- lapply(
  1:10,
  function(i) array(runif(28 * 28 * 1), dim = c(28, 28, 1))
)
x_dummy_df <- tibble::tibble(x = x_dummy_list)
y_dummy <- factor(sample(0:9, 10, replace = TRUE), levels = 0:9)
y_dummy_df <- tibble::tibble(y = y_dummy)


# Use compile_keras_grid to get the model summary
compilation_results <- compile_keras_grid(
  spec = cnn_spec,
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

