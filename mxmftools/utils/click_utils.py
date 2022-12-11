import click
import sys

# copy from https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
import click


class OptionEatAll(click.Option):
    def __init__(self, *args, **kwargs):
        self.mytype = kwargs.pop("mytype", "str")
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            for prefix in self._eat_all_parser.prefixes:
                if value[0].startswith(prefix):
                    print(self.name)
                    click.echo(f"Error: Option '{self.opts[0]}' requires an argument.")
                    sys.exit()
            done = False
            if self.mytype == "str":
                value = f"{value}"
            elif self.mytype == "strlist":
                value = [value]
            elif self.mytype == "intlist":
                value = [int(value)]
            elif self.mytype == "floatlist":
                value = [float(value)]

            while state.rargs and not done:
                for prefix in self._eat_all_parser.prefixes:
                    if state.rargs[0].startswith(prefix):
                        done = True
                if not done:
                    if self.mytype == "str":
                        value = value + " " + state.rargs.pop(0)
                    elif self.mytype == "strlist":
                        value.append(state.rargs.pop(0))
                    elif self.mytype == "intlist":
                        value.append(int(state.rargs.pop(0)))
                    elif self.mytype == "floatlist":
                        value.append(float(state.rargs.pop(0)))

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


# copy from https://stackoverflow.com/questions/40182157/shared-options-and-flags-between-commands
def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options
