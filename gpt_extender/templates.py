class ExtendTemplate:
    def __init__(self, 
                 column_name: str, 
                 new_column_name: str, 
                 context: str,
                 task: str, 
                 output: str, 
                 **kwargs
                 ) -> None:
        
        self.column_name = column_name
        self.new_column_name = new_column_name
        self.context = context
        self.task = task
        self.output = output
        self.extra_args = kwargs

    def prompt(self, text: str) -> str:
        end_msg = "Analyze this text:"
        return self.task + " " + self.output + end_msg + "\n" + text