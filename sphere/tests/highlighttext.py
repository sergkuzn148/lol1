#!/usr/bin/env python
# ANSI color coded text if the terminal supports it. No external requirements.
# Inspired by
# http://korbinin.blogspot.dk/2012/10/color-text-output-from-python.html

import sys

def highlight(string, fg, bold=False):
    '''
    Return `string` with ANSI color codes.

    :param string: String to colorize.
    :type string: str
    :param fg: Foreground text color.  Possible values: 'g', 'green', 'r', 'red',
        'y', 'yellow', 'b', 'blue', 'm' or 'magenta'
    :type fg: str
    :param bold: Bold text
    :type bold: bool

    :returns: ANSI formatted string, can be `print()`'ed directly.
    :return type: str
    '''

    attr = []
    if sys.stdout.isatty():

        if fg == 'g' or fg == 'green':
            attr.append('32')

        elif fg == 'r' or fg == 'red':
            attr.append('31')

        elif fg == 'y' or fg == 'yellow':
            attr.append('33')

        elif fg == 'b' or fg == 'blue':
            attr.append('34')

        elif fg == 'm' or fg == 'magenta':
            attr.append('35')

        else:
            raise Exception('Error: Foreground color `fg` value not understood')

        if bold:
            attr.append('1')

        return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

    else:
        return string
