import glfw
import imgui
import imgui.integrations.glfw as glfw_intergration

def main():


    # Initialize the library
    if not glfw.init():
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "Hello World", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    #glfw.make_context_current(window)

    glfw_imgui_renderer = glfw_intergration.GlfwRenderer(window)

    # initilize imgui context (see documentation)
    imgui.create_context()
    imgui.get_io().display_size = 100, 100
    imgui.get_io().fonts.get_tex_data_as_rgba32()



    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here, e.g. using pyOpenGL

        imgui.set_current_context()

        glfw_imgui_renderer.process_inputs()

        # start new frame context
        imgui.new_frame()

        # open new window context
        imgui.begin("Your first window!", True)

        # draw text label inside of current window
        imgui.text("Hello world!")

        # close current window context
        imgui.end()

        # pass all drawing comands to the rendering pipeline
        # and close frame context
        imgui.render()
        imgui.end_frame()

        glfw_imgui_renderer.render()

        # Swap front and back buffers
        #glfw.swap_buffers(window)

        # Poll for and process events
        #glfw.poll_events()

    glfw_imgui_renderer.shutdown()
    #glfw.terminate()

if __name__ == "__main__":
    main()