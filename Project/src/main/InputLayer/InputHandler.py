from __future__ import annotations



"""InputHandler singleton to pass a Data object to the Encoder layer."""

class InputHandler:    
    """Singleton class to handle input data for the Encoder layer. Still working out how we want to implement this. It might be good to have a singleton to manage the input data across the application. Then make seperate Classes for each responsibility like reading files, validating data, converting into a dataframe etc.  
    
    Prefer composition over a global singleton. Keep InputHandler lightweight and inject the specific collaborators (data loader, hyperparameter manager, dataframe). Each responsibility can live in its own class, coordinated by a small orchestrator that you instantiate explicitly. This keeps testing, concurrency, and configuration simpler than forcing everything through one process-wide instance.


    Structure it as small, explicit collaborators instead of a process-wide singleton:

    InputSource classes (e.g., CsvSource, JsonSource) handle loading and basic validation.
    SequenceBuilder converts raw records into sequential datasets.
    DataFrameAdapter turns sequences into pandas DataFrames.
    A thin InputPipeline orchestrates these components; you instantiate it with the concrete pieces per workflow.
    This keeps responsibilities isolated, eases testing, and avoids threading/config headaches linked to singletons while still giving the encoder a ready-made DataFrame.
    """

    _instance = None
    """The single instance of the InputHandler class."""


    def __new__(cls, *args, **kwargs) -> InputHandler:
        """Constructor -- Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(InputHandler, cls).__new__(cls)
        
        return cls._instance

