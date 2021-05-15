import sys
import argparse

class InputArguments:

    # ======================================
    #         DEFINITIONS FROM HERE
    # ======================================

    dnb5_file = ''
    prephase = False
    coefficient = 0.23

    # ======================================
    #               TO HERE
    # ======================================

    def __init__(self):

        pass

    def list_arguments(self):
        members = [attr for attr in dir(args) if not callable(getattr(args, attr)) and not attr.startswith("__")]
        values = [getattr(args, attr) for attr in members]
        print(members)
        print(values)

    def _list_internal_args(self):

        return [attr for attr in dir(args) if not callable(getattr(args, attr)) and not attr.startswith("__")]

    def _parse_from_arg_list(self, arg_list):
        parser = argparse.ArgumentParser(description='Process some integers.')

        parser.add_argument('dnb5',
                            type=str,
                            help='path to dnb5')

        parser.add_argument('--coefficient',
                            type=float,
                            default=self.coefficient,
                            help='sum the integers (default: find the max)')

        args = parser.parse_args(arg_list)

        list_modified_args = []
        for arg_key in self._list_internal_args():
            if arg_key in vars(args):
                prev_value = eval(f'self.{arg_key}')
                new_value = eval(f'args.{arg_key}')
                if prev_value != new_value:
                    print(f'{arg_key} updated')
                    setattr(self, arg_key, new_value)
                    list_modified_args.append((arg_key, prev_value, new_value))
                else:
                    print(f'{arg_key} unchanged')
            else:
                print(f'{arg_key} not defined internally')

        print(f'Modified arguments: {list_modified_args}')

if __name__ == '__main__':

    args = InputArguments()

    args._parse_from_arg_list(sys.argv[1:])