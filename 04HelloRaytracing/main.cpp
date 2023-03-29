#include "SampleRenderer.h"

#include <sutil/sutil.h>
#include <sutil/CUDAOutputBuffer.h>

#include <sutil/GLDisplay.h>
#include <GLFW/glfw3.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "support/stb/stb_image_write.h"

bool resize_dirty = false;
bool minimized = false;
int2 fbSize;

sutil::Trackball trackball;
sutil::Camera    camera;
bool camera_changed = true;
// Mouse state
int32_t mouse_button = -1;

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    LaunchParams* params = static_cast<LaunchParams*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->frame.fbSize.x, params->frame.fbSize.y);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->frame.fbSize.x, params->frame.fbSize.y);
        camera_changed = true;
    }
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}

static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}

static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}

static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    fbSize = make_int2(res_x, res_y);
    camera_changed = true;
    resize_dirty = true;
}

void handleCameraUpdate(SampleRenderer& renderer)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(fbSize.x) / static_cast<float>(fbSize.y));
    //camera.UVWFrame(params.U, params.V, params.W);

    renderer.setCamera(camera);
}

void displaySubframe(sutil::CUDAOutputBuffer<uint32_t>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

extern "C" int main(int ac, char** av)
{
    try {
        Model* model = loadOBJ("C:/Users/local-admin/Desktop/PRayGround/resources/model/sponza/sponza.obj");
        
        camera = sutil::Camera(/*from*/make_float3(-10.f,2.f,-12.f),
            /* at */make_float3(0.f,0.f,0.f),
            /* up */make_float3(0.f,1.f,0.f),
            35,
            1.0f);

        trackball.setCamera(&camera);
        trackball.setMoveSpeed(10.0f);
        trackball.setReferenceFrame(
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 1.0f, 0.0f)
        );
        trackball.setGimbalLock(true);

        SampleRenderer sample(model);
        sample.setCamera(camera);

        fbSize = make_int2(1200, 1024);
        sample.resize(fbSize);
        sample.render();

        std::vector<uint32_t> pixels(fbSize.x * fbSize.y);        

        GLFWwindow* window = sutil::initUI("optixPathTracer", fbSize.x, fbSize.y);
 
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetWindowSizeCallback(window, windowSizeCallback); 
        glfwSetKeyCallback(window, keyCallback);
        glfwSetWindowIconifyCallback(window, windowIconifyCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetWindowUserPointer(window, &sample.launchParams);

        //
        // Render loop
        //
        {
            sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

            sutil::CUDAOutputBuffer<uint32_t> output_buffer(
                output_buffer_type,
                fbSize.x,
                fbSize.y
            );

            output_buffer.setStream(sample.stream);
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time(0.0);
            std::chrono::duration<double> render_time(0.0);
            std::chrono::duration<double> display_time(0.0);

            do
            {
                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                if (resize_dirty){
                    sample.resize(fbSize);         
                    output_buffer.resize(fbSize.x, fbSize.y);
                    resize_dirty = false;
                }
                handleCameraUpdate(sample);

                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                sample.render(output_buffer);
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe(output_buffer, gl_display, window);
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                sutil::displayStats(state_update_time, render_time, display_time);

                glfwSwapBuffers(window);

            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
        }

        sutil::cleanupUI(window);

        /*sample.downloadPixels(pixels.data());

        sutil::ImageBuffer buffer;
        buffer.data = pixels.data();
        buffer.width = fbSize.x;
        buffer.height = fbSize.y;
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        sutil::displayBufferWindow(*av, buffer);

        const std::string fileName = "osc_example2.png";
        stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
            pixels.data(), fbSize.x * sizeof(uint32_t));
        std::cout 
            << std::endl
            << "Image rendered, and saved to " << fileName << " ... done." << std::endl            
            << std::endl;*/
    }
    catch (std::runtime_error& e) {
        std::cout  << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}