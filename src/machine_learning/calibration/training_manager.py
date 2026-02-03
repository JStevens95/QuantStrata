import os
import time
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from src.machine_learning.models.gnn_rnn_hybrid.model import HybridGnnRnn

# define logging at module level
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfiguration:
    # model identification
    name: str | None = None
    mode: str = 'default'
    model: str | None = None
    # training parameters
    epochs: int | None = None
    loss: Any |None = None
    learning_rate: float = 0.0010
    batch_size: int | None = None
    patience: int = 100
    optimizer: Any | None = None
    metrics: List[str] = field(default_factory=lambda: ['mae', 'mse'])
    shuffle: bool = False
    pretrain: bool = False
    run_eagerly: bool = False
    verbose: int = 1
    # callback parameters
    save_best_only: bool = True
    save_weights_only: bool = True
    early_stopping: bool = True
    min_lr: float = 0.0001
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float | None = None
    reduce_lr_patience: float | None = None
    # folder paths.
    model_dir: str | None = None


class TrainingManager:
    """
    Manager for training and evaluating machine learning models.

    Supported models: ['HybridGnnRnn']
    """

    def __init__(
            self, training_ds: tf.data.Dataset, model_config: Dict[str, Any],
            validation_ds: tf.data.Dataset = None, tf_strategy: Optional[Any] = None,
            custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> None:
        """
        Initiate the training manager.

        :param training_ds: tensorflow dataset for training data
        :param model_config: dictionary configuration for machine learning model.
        :param validation_ds: tensorflow dataset for validation data.
        :param tf_strategy: optional distribution strategy
        :param custom_callbacks: optional list of Keras callbacks to add to every training stage
        """
        # initiate required variables
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.model_config = model_config
        self.tf_strategy = tf_strategy if tf_strategy else tf.distribute.get_strategy()
        self.custom_callbacks = custom_callbacks or []

        # initiate derived variables.
        self.model: tf.keras.Model | None = None
        self.training_history: Dict[str, Any] = {}
        self.training_metadata: Dict[str, Any] = {}

    def build_callbacks(self, stage: TrainingConfiguration) -> List[Any]:
        """
        Build callbacks based on training configuration.

        :param stage: custom training stage
        :return:
        """
        # 0. define list to hold training callbacks & paths.
        callbacks = []
        ext = '.weights.h5' if stage.save_weights_only else '.keras'
        checkpoint_path = os.path.join(stage.model_dir, stage.model, 'mlearn',
                                       f'{stage.name}_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}{ext}')
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        # 1. define model checkpoint callback.
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, monitor='val_loss' if self.validation_ds is not None else 'loss',
                save_best_only=stage.save_best_only, verbose=stage.verbose, save_weights_only=stage.save_weights_only
            )
        )

        # 2. define early stopping callback
        if stage.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if self.validation_ds is not None else 'loss', patience=stage.patience,
                    restore_best_weights=True, verbose=stage.verbose
                )
            )

        # 3. define reducing learning callback.
        if stage.reduce_lr_on_plateau:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if self.validation_ds is not None else 'loss',
                    factor=stage.reduce_lr_factor if stage.reduce_lr_factor is not None else 0.5,
                    patience=stage.reduce_lr_patience if stage.reduce_lr_patience is not None else 10,
                    mode='min', min_lr=stage.min_lr
                )
            )
        callbacks.extend(self.custom_callbacks)
        return callbacks

    def build_model(self, stage: TrainingConfiguration) -> Union[HybridGnnRnn]:
        """
        Build model instance for specified machine learning model.

        :param stage: custom training stage
        :return:
        """
        if stage.model == "HybridGnnRnn":
            return HybridGnnRnn(model_config=self.model_config, name=stage.name)
        else:
            raise ValueError(f"Unsupported model type: {stage.model}")

    def evaluate_training(self):
        """
        Evaluate training performance.

        Plot training analytics:
             - training / validation learning curves.
             - learning rate per epoch.
             - gradient norms
             - wall-clock time / epoch
        :return:
        """

    def evaluate_model(self):
        """
        Evaluate trained model performance.

        Plot training analytics:
            - training / validation model performance.
            -
            - validation residual distributions.
        :return:
        """


    def run(self, stages: List[TrainingConfiguration]):
        """
        Run multiple training stages either default (1 stage) or multi-stage in sequence.

        :param stages: list of custom training stage(s)
        :return:
        """
        # run different training configuration stages.
        for stage in stages:
            self.train_stage(stage)

    def train_stage(self, stage: TrainingConfiguration) -> None:
        """

        :param stage: custom training stage
        :return:
        """
        # 0. validate input data.
        assert isinstance(self.training_ds, tf.data.Dataset), "training dataset must be a Tensorflow Dataset"
        if self.validation_ds is not None and not isinstance(self.validation_ds, tf.data.Dataset):
            raise ValueError("validation dataset must be a Tensorflow Dataset")

        # 1. train the model.
        st = time.time(); logger.info(f"Starting training for {stage.model}; epochs={stage.epochs}")

        # 2. build model instance and set mode.
        with self.tf_strategy.scope():
            # build model
            self.model = self.build_model(stage)

            # ensure model is built correctly.

            # compile the model.
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=stage.learning_rate), run_eagerly=stage.run_eagerly,
                loss=stage.loss, metrics=stage.metrics
            )

        # 3. build callbacks.
        callbacks = self.build_callbacks(stage)

        # 4. fit the model.
        stage_history = self.model.fit(
            self.training_ds, epochs=stage.epochs, callbacks=callbacks, verbose=stage.verbose,
            validation_data=self.training_ds if stage.pretrain else self.validation_ds
        )

        # 5. log model training results.
        train_time = time.time() - st
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info(f"{stage.model.upper()} final training loss {stage_history.history['loss'][-1]:.5f}")
        logger.info(f"{stage.model.upper()} final validation loss {stage_history.history['val_loss'][-1]:.5f}")

        # 6. add stage history.
        self.training_history[stage.name] = stage_history.history

        # 7. add stage metadata.
        self.training_metadata[stage.name] = {
            'model_name': self.model.name, 'model_type': stage.model, 'epochs': stage.epochs, 'patience': stage.patience,
            'training_time': train_time, 'final_epoch': len(stage_history.history['loss']),
            'final_loss': float(stage_history.history['loss'][-1]),
            'final_val_loss': float(stage_history.history['val_loss'][-1]),
            'optimizer': self.model.optimizer.get_config()
        }

    @staticmethod
    def _extract_targets(ds: tf.data.Dataset):
        """
        Extract targets from tensorflow dataset.

        :param ds: tensorflow dataset
        :return:
        """
        target_ds = ds.map(lambda inputs, targets: targets)
        targets_list = list(target_ds.unbatch().as_numpy_iterator())
        return tf.stack(targets_list, axis=0)

    @staticmethod
    def _plot_learning_curves(
            model: Union[HybridGnnRnn], stage: TrainingConfiguration, save_dir: Optional[str] = None
    ) -> None:
        """
        Plot learning curves (training & validation loss) for a fitted keras model.

        :param model: fitted keras model.
        :param save_dir:
        :return:
        """
        # if save dir provided, check path exists.
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # loss curve figure.
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(model.history.epoch, model.history.history['loss'], label='training')
        ax1.plot(model.history.epoch, model.history.history['val_loss'], label='validation')
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title(f"Learning Curves: {stage.model} | Epochs: {stage.epochs} | Batch Size: {stage.batch_size}")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True)
        plt.tight_layout()

        # save plot if specified, show on screen if not.
        if save_dir:
            fig1.savefig(os.path.join(save_dir, f"{stage.name}_learning_curves.png"), dpi=200, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig1)