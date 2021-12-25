from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, List, Tuple, Union

import pygame

ColorInput = Union[
    pygame.Color, str, List[int], Tuple[int,
                                        int, int], Tuple[int, int, int, int]
]

RgbaOutput = Tuple[int, int, int, int]


def dummy(cv: Canvas):  # pylint: disable=unused-argument
    pass


class Canvas:

    def __init__(self, screenSize: Tuple[int, int] = (512, 512), title="Graphics Window"):
        # initialize all pygame modules
        pygame.init()  # pylint: disable=no-member
        # indicate rendering details
        displayFlags = pygame.GL_DOUBLEBUFFER  # pylint: disable=no-member
        # initialize buffers to perform antialiasing
        pygame.display.gl_set_attribute(
            pygame.GL_MULTISAMPLEBUFFERS, 1)  # pylint: disable=no-member
        pygame.display.gl_set_attribute(
            pygame.GL_MULTISAMPLESAMPLES, 4)  # pylint: disable=no-member
        # use a core OpenGL profile for cross-platform compatibility
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)  # pylint: disable=no-member
        # create and display the window
        self._screen: pygame.Surface = pygame.display.set_mode(
            screenSize, displayFlags)
        # set the text that appears in the title bar of the window
        pygame.display.set_caption(title)

        # determine if main loop is active
        self._running: bool = True
        # manage time-related data and operations
        self._clock: pygame.time.Clock = pygame.time.Clock()
        self._update: Callable[[Canvas], None] = dummy

    @property
    def width(self) -> int:
        return self._screen.get_width()

    @property
    def height(self) -> int:
        return self._screen.get_height()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def surface(self):
        return self._screen

    def stop(self):
        self._running = False

    def initialize(self):
        pass

    def set_update(self, updatefn: Callable[[Canvas], None]):
        self._update = updatefn

    def run(self):
        ## startup ##
        self.initialize()
        ## main loop ##
        while self._running:
            ## process input ##
            # iterate over all user input events (such as keyboard or
            # mouse)that occurred since the last time events were checked
            for event in pygame.event.get():
                # quit event occurs by clicking button to close window
                if event.type == pygame.QUIT:  # pylint: disable=no-member
                    self._running = False

            ## update ##
            self._update(self)
            ## render ##
            # display image on screen
            pygame.display.flip()

            # pause if necessary to achieve 60 FPS
            self._clock.tick(60)

        ## shutdown ##
        pygame.quit()  # pylint: disable=no-member

    def set_pixel(self, x: int, y: int, color: ColorInput):
        self._screen.set_at((x, y), color)

    def set_pixelf(self, x: int, y: int, color: Tuple[float, float, float]):
        new_color = (min(int(color[0] * 255), 255),
                     min(int(color[1] * 255), 255), min(int(color[2] * 255), 255))
        self._screen.set_at((x, y), new_color)

    def get_pixel(self, x: int, y: int) -> RgbaOutput:
        return self._screen.get_at((x, y))

    def get_pixel_array(self) -> pygame.PixelArray:
        return pygame.PixelArray(self._screen)

    @contextmanager
    def get_pixel_array_cm(self):
        px_array: pygame.PixelArray = pygame.PixelArray(self._screen)
        try:
            yield pygame.PixelArray(self._screen)
        finally:
            px_array.close()

    def to_img(self, file: str):
        pygame.image.save(self._screen, file)


class CanvasImg(Canvas):
    def __init__(self, file: str, screenSize: Tuple[int, int] = (512, 512)):
        super().__init__(screenSize)
        self._file: str = file

    def run(self):
        def dummy_func(cv: CanvasImg):
            cv.to_img(self._file)
            cv.stop()

        self._update = dummy_func
        super().run()
