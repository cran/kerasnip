## ----include = FALSE----------------------------------------------------------
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

## ----data-prep----------------------------------------------------------------
# Split data into training and testing sets
set.seed(123)
iris_split <- initial_split(iris, prop = 0.8, strata = Species)
iris_train <- training(iris_split)
iris_test <- testing(iris_split)

# Create cross-validation folds for tuning
iris_folds <- vfold_cv(iris_train, v = 3, strata = Species)

## ----define-kerasnip-model----------------------------------------------------
# Define layer blocks
input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}
dense_block <- function(model, units = 10) {
  model |> layer_dense(units = units, activation = "relu")
}
output_block <- function(model, num_classes) {
  model |> layer_dense(units = num_classes, activation = "softmax")
}

# Create the kerasnip model specification function
create_keras_sequential_spec(
  model_name = "iris_mlp",
  layer_blocks = list(
    input = input_block,
    dense = dense_block,
    output = output_block
  ),
  mode = "classification"
)

## ----define-tune-spec---------------------------------------------------------
# Define the tunable model specification
tune_spec <- iris_mlp(
  dense_units = 16, # Keep architecture fixed for this example
  fit_epochs = tune(),
  fit_batch_size = tune(),
  compile_optimizer = tune(),
  compile_loss = tune(),
  learn_rate = tune()
) |>
  set_engine("keras")

print(tune_spec)

## ----create-workflow-grid-----------------------------------------------------
# Create a simple recipe
iris_recipe <- recipe(Species ~ ., data = iris_train) |>
  step_normalize(all_numeric_predictors())

# Create the workflow
tune_wf <- workflow() |>
  add_recipe(iris_recipe) |>
  add_model(tune_spec)

# Define the tuning grid
params <- extract_parameter_set_dials(tune_wf) |>
  update(
    fit_epochs = epochs(c(10, 30)),
    fit_batch_size = batch_size(c(16, 64), trans = NULL),
    compile_optimizer = optimizer_function(values = c("adam", "sgd", "rmsprop")),
    compile_loss = loss_function_keras(values = c("categorical_crossentropy", "kl_divergence")),
    learn_rate = learn_rate(c(0.001, 0.01), trans = NULL)
  )

set.seed(456)
tuning_grid <- grid_regular(params, levels = 2)

tuning_grid

## ----tune-model, cache=TRUE---------------------------------------------------
tune_res <- tune_grid(
  tune_wf,
  resamples = iris_folds,
  grid = tuning_grid,
  metrics = metric_set(accuracy, roc_auc),
  control = control_grid(save_pred = FALSE, save_workflow = TRUE, verbose = FALSE)
)

## ----inspect-results----------------------------------------------------------
# Show the best performing models based on accuracy
show_best(tune_res, metric = "accuracy")

# Plot the results
autoplot(tune_res) + theme_minimal()

# Select the best hyperparameters
best_params <- select_best(tune_res, metric = "accuracy")
print(best_params)

## ----finalize-fit-------------------------------------------------------------
# Finalize the workflow
final_wf <- finalize_workflow(tune_wf, best_params)

# Fit the final model
final_fit <- fit(final_wf, data = iris_train)

print(final_fit)

## ----predict------------------------------------------------------------------
# Make predictions
predictions <- predict(final_fit, new_data = iris_test)

# Evaluate performance
bind_cols(predictions, iris_test) |>
  accuracy(truth = Species, estimate = .pred_class)

