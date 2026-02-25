"""Abstract base class for pattern storage models."""

from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for models that learn and represent discretised patterns.

    A model learns patterns from discretised sequence data (represented as chains
    of nodes) and can be used to check whether new data matches learned patterns
    or to update the model with new observations.

    Subclasses must implement methods for learning patterns, updating the model with
    new data, and checking pattern membership.
    """

    @abstractmethod
    def update(self, data: list) -> list:
        """
        Update the model with a new data point.

        Incrementally updates the model's learned patterns with a new observation,
        adjusting the model state accordingly.

        Parameters
        ----------
        data : list
            New data to incorporate into the model.

        Returns
        -------
        list
            Updated model state or resulting patterns after incorporating the new data.
        """
        pass

    @abstractmethod
    def learn(self, data: list) -> list:
        """
        Learn patterns from the provided dataset.

        Analyses the provided dataset to extract and store patterns that characterise
        the data. This is typically used for batch learning from a complete dataset.

        Parameters
        ----------
        data : list
            Dataset to learn patterns from.

        Returns
        -------
        list
            Learned patterns or model representation derived from the data.
        """
        pass

    @abstractmethod
    def contains(self, data: list) -> bool:
        """
        Check if the model contains the given data.

        Determines whether the provided data matches any of the patterns learned
        by the model, indicating membership or recognition.

        Parameters
        ----------
        data : list
            Data to check for pattern membership.

        Returns
        -------
        bool
            True if the data matches learned patterns, False otherwise.
        """
        pass
