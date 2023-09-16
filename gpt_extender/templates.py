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
        msg_ending = "Analyze this text:"
        msg_order = [self.task, self.output, msg_ending, text]
        return "\n".join(msg_order)
    
    def prompt_synthetic(self, text: str, output_size: int) -> str:
        msg_ending = "Based on the following text, provide {output_size} examples:"
        msg_order = [self.context, self.task, self.output, msg_ending.format(output_size=output_size), text]
        return  "\n".join(msg_order)