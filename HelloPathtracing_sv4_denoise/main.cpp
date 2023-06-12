#pragma once 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "support/stb/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "support/stb/stb_image.h"

#include "SimplePathtracer.h"
#include "Probe.h"

#include <sutil/GLDisplay.h>
#include <sutil/Trackball.h>
#include <GLFW/glfw3.h>

// bm, for saving data files, currently in SUTIL.cpp to save fps
// TODO: better data writing
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// scenes
//#define CRYTEK_SPONZA
//#define SAN_MIGUEL
//#define LOST_EMPIRE

bool resize_dirty = false;
bool minimized = false;
int2 fbSize;
sutil::Trackball trackball;
sutil::Camera    camera;
bool camera_changed = true;// no effect

// Mouse state
int32_t mouse_button = -1;

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    LaunchParams* params = static_cast<LaunchParams*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->frame.size.x, params->frame.size.y);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->frame.size.x, params->frame.size.y);
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
    else if (key == GLFW_KEY_S)
    {
        //!TODO: save rendered image file
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

//! bm: no effect this function, isn't it redundant?, i keep commented 
// currently no use, so skip
//void initLaunchParams(SampleRenderer& pathtracer) {
//    LaunchParams& params = pathtracer.launchParams;
//
//     params.samples_per_launch = 4; 
//    params.frame.subframe_index = 0u;
//
//    const float light_size = 200.f;
//
//    params.light.emission = make_float3(15.0f, 15.0f, 15.0f);
//    params.light.corner = make_float3(-1000 - light_size, 1200, -light_size);
//    params.light.v1 = make_float3(2.f * light_size, 0, 0);
//    params.light.v2 = make_float3(0, 0, 2.f * light_size);        
//    params.light.normal = normalize(cross(params.light.v1, params.light.v2));
//}

void loadProbe(ProbeData& probe, std::string hdrFile) {
    int resX, resY, channel;
    float4* data = (float4*)stbi_loadf(hdrFile.c_str(), &resX, &resY, &channel, 4);

    probe.width = resX;
    probe.height = resY;
    int numPixels = resX * resY;
    probe.data = data;

    probe.BuildCDF();
}

//!bm: modified with solid background 
void loadColor(ProbeData& probe, float3 color){
    //!TODO: bm, should change according to fb size, currently not working fbSize.x
    int resX = 3840;// fbSize.x;//1200; 
    int resY = 2160;//fbSize.y;//1024;
	int numPixels = resX * resY;
	float4* data = new float4[numPixels];
    for (int i = 0; i < numPixels; i++) {
		data[i] = make_float4(color.x, color.y, color.z, 1.0f);
	}
	probe.width = resX;
	probe.height = resY;
	probe.data = data;
	probe.BuildCDF();
}

extern "C" int main(int ac, char** av)
{
    try {
// Test Models

#if defined(CRYTEK_SPONZA)
    Model* model = loadOBJ("C:/Users/local-admin/Desktop/PRayGround/resources/model/crytek_sponza/sponza.obj");
#elif defined(SAN_MIGUEL)    
    Model* model = loadOBJ("C:/Users/local-admin/Desktop/PRayGround/resources/model/San_Miguel/san-miguel.obj");
#elif defined(LOST_EMPIRE)
    Model* model = loadOBJ("C:/Users/local-admin/Desktop/PRayGround/resources/model/lost-empire/lost_empire.obj");
#else
    Model* model = loadOBJ("C:/Users/local-admin/Desktop/FovTiX/G3D_data10/research/model/rungholt/rungholt/rungholt.obj");
#endif
/**
* @environment lighting 
*/
#define ENV_LIGHT_OFF
#if defined(ENV_LIGHT_ON)
        ProbeData probe;
        loadProbe(probe, "C:/Users/local-admin/Desktop/PRayGround/resources/image/garden_nook_8k.hdr");
#else        
    //!bm: solid color background, white color (change the number for light intensity in between 1.0 to any other )
    ProbeData probe;
    loadColor(probe, make_float3(1.0f)); 

#endif

//!--------------------------------------------------------- bm: camera inputs 
#if defined(CRYTEK_SPONZA)
//! for crytek-sponza     
     camera = sutil::Camera(/*from*/make_float3(-1293.07f, 154.681f, 1.0f),
            /* at */make_float3(0.f,200.f,0.f),
            /* up */make_float3(0.f,1.f,0.f),
            35,
            1.0f);
#elif defined(SAN_MIGUEL)
//! for san-miguel
     camera = sutil::Camera(make_float3(26, 8, -2),
            make_float3(0.f,0.f,0.f),
            make_float3(0.f,1.f,0.f),
            35,
            1.0f);
#elif defined(LOST_EMPIRE)
     // for lost empire
     camera = sutil::Camera(make_float3(-70, 40, 100),
         make_float3(0.f, 0.f, 0.f),
         make_float3(0.f, 1.f, 0.f),
         45,
         1.0f);

#else
// for rungholt
        camera = sutil::Camera(make_float3(-70, 40, 100),
            make_float3(0.f, 0.f, 0.f),
            make_float3(0.f, 1.f, 0.f),
            45,
            1.0f);
#endif 
//--------------------------------------------------------------------
     // camera parameters 
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

//! -----------------------------------------------------WINDOW HANDLING 
        fbSize = make_int2(3840, 2160);
        //!TODO:
        //sample.launchParams.viewportSize.x, sample.launchParams.viewportSize.y); 
        // 3840,2160);//1200,1024
        // testing
       // std::cout << "frame buffer size: " << sample.launchParams.viewportSize.x << '\t' << sample.launchParams.viewportSize.y << '\n';
        sample.resize(fbSize);
        //testing
//        std::cout << "frame buffer size: " << sample.launchParams.viewportSize.x << '\t' << sample.launchParams.viewportSize.y << '\n';

        //bm, commented, no effect!!! It is tied with manual light import        
        //initLaunchParams(sample);

        sample.setProbe(probe);// probability

        //%bm  render()
        // becaue it is being called in renderin loop, so it is not needed here
        //sample.render();
        std::vector<uint32_t> pixels(fbSize.x * fbSize.y);        

        GLFWwindow* window = sutil::initUI("V4", fbSize.x, fbSize.y);
 
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetWindowSizeCallback(window, windowSizeCallback); 
        glfwSetKeyCallback(window, keyCallback);
        glfwSetWindowIconifyCallback(window, windowIconifyCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetWindowUserPointer(window, &sample.launchParams);
        //
//!---------------------------------------------------------- Render loop
        // important for buffer handling, visualization, gpu 
        {
            sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
            //bm: for nvlink configuration, check, currently no support 
            // 0 = 190 mx, 1 = 65 mx, 2 = 90 mx, 3 = 65 mx, for sponza
            // CUDA_DEVICE is giving the best frame-rate 

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
                        
            CUDA_SYNC_CHECK();
            do
            {   //BM: gaze points 
                double cposx;
                double cposy;
                glfwGetCursorPos(window, &cposx, &cposy);
                sample.launchParams.frame.c.x = cposx;
                sample.launchParams.frame.c.y = cposy;

                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                if (camera_changed || resize_dirty)
                    sample.launchParams.frame.subframe_index = 0;

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

//--------------------------------------------------------------------
                //!TODO: better imgui UI              

                //sutil::displayStats( state_update_time, render_time, display_time, cposx, cposy/*, sample.launchParams.frame.subframe_index*/);


                //std::vector<double>stateUpdateTime; 
                //std::vector<double>renderTime;  
                //std::vector<double>displayTime; 
                //std::vector<double>positionX; 
                //std::vector<double>positionY; 
                //stateUpdateTime.push_back(state_update_time.count());
                //renderTime.push_back(render_time.count());
                //displayTime.push_back(display_time.count());
                //positionX.push_back(cposx); 
                //positionY.push_back(cposy); 
//// bm: write to file
//#define DATA_FORMAT_DAT
//#ifdef DATA_FORMAT_DAT
//                std::ofstream file1("../../HelloPathtracing_sv4_denoise/lost_empire_uniform.tsv", std::ios::app);
//                if (!file1) {
//                    					std::cout << "Error opening file" << std::endl;
//					return 1;
//                }
//                else {
//                    for (int i = 0; i < 1000 /*stateUpdateTime.size()*/; ++i) {
//						file1 << /* i << '\t' << */ stateUpdateTime[i] << '\t' << renderTime[i] << '\t' << displayTime[i] << '\t' << positionX[i] << '\t' << positionY[i] << '\n';
//                        
//					}
//				}
//                file1.close();
//#else
//                std::ofstream file2("../../HelloPathtracing_sv4_denoise/data.", std::ios::app);
//                // problem: csv all data in single cell
//                if(!file2){
//					std::cout << "Error opening file" << std::endl;
//					return 1;
//				}
//                else {
//                    for (int i = 0; i < stateUpdateTime.size(); i++) {
//                        file2 << i << ',' << stateUpdateTime[i] << ',' << renderTime[i] << ',' << displayTime[i] << ',' << positionX[i] << ',' << positionY[i] << '\n';
//                    }
//                }
//                file2.close();
//#endif
//--------------------------------------------------------------------
                 glfwSwapBuffers(window);

                sample.launchParams.frame.subframe_index += 1;

            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
        }
        uint32_t* pixels_ = (uint32_t*)malloc(fbSize.x * fbSize.y * sizeof(int));
        
        //!TODO: saving frame machanism 
        //sample.downloadPixels(pixels_);

        //------------------------- sv: unncecessary , need to clean up
        //sutil::ImageBuffer buffer;
        //buffer.data = pixels.data();
        //buffer.width = fbSize.x;
        //buffer.height = fbSize.y;
        //buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        //sutil::displayBufferWindow(*av, buffer);
        //------------------------- end unncesssary

        //!TODO: saving frame machanism 
        //const std::string fileName = "../../HelloPathtracing_sv4_denoise/test_saving.png";
        //stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
        //    (void*)pixels_, fbSize.x * sizeof(uint32_t));
        //std::cout
        //    << std::endl
        //    << "Image rendered, and saved to " << fileName << " ... done." << std::endl
        //    << std::endl;

        sutil::cleanupUI(window);
    }
    catch (std::runtime_error& e) {
        std::cout  << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}