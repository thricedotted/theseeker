# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import random

noise_chars = ["♠♣♥♦", 
#"♩♪♫♬", 
#"✗✓",
"↓←↑→",
"†‡",
"1234567890",
"01",
"xo",
"@#",
":;'\"",
"/%"
]

compatible_characters = ["†‡†‡",
"↓←↑→",
"↑→↓←",
"=≠=≠",
"-=-=",
"=|=|",
"=/=\\",
"-/-\\",
".].[",
".).(",
"????",
"@*@*",
"?*?*"
#"⊤⊣⊥⊢",
#"⋯⋮⋯⋮"
#"-⊶-⊷",
]

# resolution: 80x60

#box_width = 30
#box_height = 10
#border_width = 2

def make_frame(box_width=30, box_height=10, 
               border_vertical=2, border_horizontal=2):
    # TODO: shouldn't hardcode this in idc
    #box_width = random.randint(15, 30) * 2
    #box_height = random.randint(10, 30)
    #border_vertical = random.randint(2, 4)
    #border_horizontal = random.randint(2, 5)

    chars = random.choice(compatible_characters)
    if random.random() < 0.5: reversed(chars)
    top, left, bottom, right = chars

    top_border = top * (box_width + border_horizontal * 2)
    bottom_border = bottom * (box_width + border_horizontal * 2)

    right_border = right * (border_horizontal)
    left_border = left * (border_horizontal)

    # build lines for borders
    side_borders = "{left_border}{empty}{right_border}".format(
                        right_border=right_border,
                        left_border=left_border, 
                        empty=(" " * box_width))

    # stitch together
    frame_lines = '\n'.join(('\n'.join(top_border for _ in range(border_vertical)),
                             '\n'.join(side_borders for _ in range(box_height)),
                             '\n'.join(bottom_border for _ in range(border_vertical))
                           ))

    return frame_lines

def make_page_frame():
    box_width = random.randint(15, 30) * 2
    box_height = random.randint(5, 15) * 2

    border_vertical = (60 - box_height) / 2
    border_horizontal = (80 - box_width) / 2

    return make_frame(box_width, box_height, border_vertical, border_horizontal)
    

#print(make_frame(box_width, box_height, 
#                 border_vertical, border_horizontal))

#frame = make_frame(box_width, box_height, 
#                 border_vertical, border_horizontal)
#text = "WOW\nWHAT\nA\nBASKETBALL\nno"

page = '\n'.join(''.join(' ' for _ in range(80)) for _ in range(60))

def insert_text(frame, text):
    height = text.count('\n') // 2 + 1
    median = frame.count('\n') // 2 + 1

    text_lines = text.splitlines()
    frame_lines = frame.splitlines()

    width = len(frame_lines[0])

    for i, line in enumerate(frame_lines[median-height:median+height]):
        try:
            idx = (median-height) + i
            text_line = "{:^{width}}".format(text_lines[i], width=width)
            frame_lines[idx] = ''.join(t if t != ' ' else c if c != ' ' else ' ' \
                                 for t, c in zip(line, text_line))
        except IndexError:
            break
        
    return '\n'.join(frame_lines)

def insert_into_page(text):
    return insert_text(page, text)

#print(insert_text(frame, text))



def page_noise(rate=0.003, width=80, height=60):
    chars = random.choice(noise_chars)
    return '\n'.join(
                    ''.join(random.choice(chars) if random.random() < rate else ' ' 
                    for _ in range(width))
                for _ in range(height))

