////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <cmath>

#define GLEW_STATIC
#include "GL/glew.h"
#include "GL/glfw3.h"

#include "arcball.h"
#include "cvec.h"
#include "geometrymaker.h"
#include "glsupport.h"
#include "matrix4.h"
#include "ppm.h"
#include "rigtform.h"

#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"

#include "sgutils.h"
#include "renderstates.h"
#include "geometry.h"
#include "uniforms.h"

#include "mesh.h"

using namespace std;

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.5.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.5. Use GLSL 1.5 unless your system does not support it.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = false;

static const float g_frustMinFov = 60.0; // A minimal of 60 degree field of view
static float g_frustFovY =
    g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;  // near plane
static const float g_frustFar = -50.0;  // far plane
static const float g_groundY = -2.0;    // y coordinate of the ground
static const float g_groundSize = 10.0; // half the ground length

static GLFWwindow *g_window;

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static double g_wScale = 1;
static double g_hScale = 1;

//enum ObjId { SKY = 0, OBJECT0 = 1, OBJECT1 = 2 };
enum SkyMode
{
    WORLD_SKY = 0,
    SKY_SKY = 1
};

static const char *const g_objNames[] = {"Sky", "Ground", "Robot1", "Robot2"};

static bool g_mouseClickDown = false; // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false;            // space state, for middle mouse emulation
static double g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;

static int g_activeEyeIndex = 0;
static shared_ptr<SgRbtNode> g_activeEye;
static SkyMode g_activeCameraFrame = WORLD_SKY;

static bool g_displayArcball = true;
static double g_arcballScreenRadius = 40; // number of pixels
static double g_arcballScale = 1;

static list<vector<RigTForm> > g_keyFrames;
static list<vector<RigTForm> >::iterator g_currentFrame;
static vector<shared_ptr<SgRbtNode> > g_tempFrame;

static int g_framesPerSecond = 60;      // Frames to render per second
static int g_msBetweenKeyFrames = 2000; // 2 seconds between keyframes
static double g_lastFrameClock;
static bool g_playingAnimation = false; // Is the animation playing?
static int g_animateTime = 0;           // Time since last key frame
static bool g_picking = false;

static bool g_shellNeedsUpdate = false;
static bool g_smoothSubdRendering = false;

// --------- Materials
// This should replace all the contents in the Shaders section, e.g., g_numShaders, g_shaderFiles, and so on
static shared_ptr<Material> g_redDiffuseMat,
g_blueDiffuseMat,
g_bumpFloorMat,
g_arcballMat,
g_pickingMat,
g_lightMat;

shared_ptr<Material> g_overridingMaterial;


static shared_ptr<Material> g_bunnyMat; // for the bunny

static vector<shared_ptr<Material>> g_bunnyShellMats; // for bunny shells

// New Geometry
static const int g_numShells = 24; // constants defining how many layers of shells
static double g_furHeight = 0.21;
static double g_hairyness = 0.7;

static shared_ptr<SimpleGeometryPN> g_bunnyGeometry;
static vector<shared_ptr<SimpleGeometryPNX>> g_bunnyShellGeometries;
static Mesh g_bunnyMesh;

// New Scene node
static shared_ptr<SgRbtNode> g_bunnyNode;


// --------- Geometry



// typedef that declares our own Shape node which draws using our Geometry.
typedef SgGeometryShapeNode MyShapeNode;

// Declare the scene graph and pointers to suitable nodes in the scene
// graph
static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node, g_light1Node, g_light2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode; // used later when you do picking
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

static shared_ptr<SimpleGeometryPN> g_meshGeometry;
static Mesh g_mesh;
static int g_subdLevels = 0;

// Global variables for used physical simulation
static const Cvec3 g_gravity(0, -0.5, 0); // gavity vector
static double g_timeStep = 0.02;
static double g_numStepsPerFrame = 10;
static double g_damping = 0.96;
static double g_stiffness = 4;
static int g_simulationsPerSecond = 60;

static std::vector<Cvec3>
    g_tipPos,      // should be hair tip pos in world-space coordinates
    g_tipVelocity; // should be hair tip velocity in world-space

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0),
    g_light2(-2, -3.0, -5.0); // define two lights positions in world space
static RigTForm g_skyRbt = RigTForm(Cvec3(0.0, 0.25, 4.0));
static RigTForm g_objectRbt[2] = {RigTForm(Cvec3(-1, 0, 0)),
                                  RigTForm(Cvec3(1, 0, 0))};
static Cvec3f g_objectColors[2] = {Cvec3f(1, 0, 0), Cvec3f(0, 0, 1)};

///////////////// END OF G L O B A L S
/////////////////////////////////////////////////////

static void initGround()
{
    int ibLen, vbLen;
    getPlaneVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makePlane(g_groundSize * 2, vtx.begin(), idx.begin());
    g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes()
{
    int ibLen, vbLen;
    getCubeVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makeCube(1, vtx.begin(), idx.begin());
    g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere()
{
    int ibLen, vbLen;
    getSphereVbIbLen(20, 10, vbLen, ibLen);

    // Temporary storage for sphere Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    makeSphere(1, 20, 10, vtx.begin(), idx.begin());
    g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

static void initNormals(Mesh &m)
{
    for (int i = 0; i < m.getNumVertices(); ++i)
    {
        m.getVertex(i).setNormal(Cvec3(0));
    }
    for (int i = 0; i < m.getNumFaces(); ++i)
    {
        const Mesh::Face f = m.getFace(i);
        const Cvec3 n = f.getNormal();
        for (int j = 0; j < f.getNumVertices(); ++j)
        {
            f.getVertex(j).setNormal(f.getVertex(j).getNormal() + n);
        }
    }
    for (int i = 0; i < m.getNumVertices(); ++i)
    {
        Cvec3 n = m.getVertex(i).getNormal();
        if (norm2(n) > CS175_EPS2)
            m.getVertex(i).setNormal(normalize(n));
    }
}

static VertexPN getVertexPN(Mesh& m, const int face, const int vertex) {
    const Mesh::Face f = m.getFace(face);
    const Cvec3 n = g_smoothSubdRendering ? f.getVertex(vertex).getNormal() : f.getNormal();
    const Cvec3& v = f.getVertex(vertex).getPosition();
    return VertexPN(v[0], v[1], v[2], n[0], n[1], n[2]);
}


// Specifying shell geometries based on g_tipPos, g_furHeight, and g_numShells.
// You need to call this function whenver the shell needs to be updated
static void updateShellGeometry()
{
    vector<VertexPNX> verts;
    verts.reserve(g_bunnyMesh.getNumFaces() * 6); // conservative estimate of num of vertices
    Cvec3 texture;
    Matrix4 worldToBunny = rigTFormToMatrix(getPathAccumRbt(g_world, g_bunnyNode, -1));

    for (int i = 0; i < g_numShells; ++i)
    {
        for (int j = 0; j < g_bunnyMesh.getNumFaces(); ++j)
        {
            Mesh::Face f = g_bunnyMesh.getFace(j);

            for (int k = 0; k < f.getNumVertices(); ++k)
            {
                Cvec3 nor = f.getVertex(k).getNormal();
                Cvec3 p = f.getVertex(k).getPosition();
                Cvec3 s = p + (nor * g_furHeight);
                Cvec3 t = g_tipPos[f.getVertex(k).getIndex()];
                Cvec3 n = nor * (g_furHeight / g_numShells);

                Cvec3 d = (t - p - (n * g_numShells)) / (g_numShells * g_numShells);

                Cvec3 position = p + n * i + d * ((i * i) - i);
                Cvec3 normal = n + d * (i - 1);

                if (k == 0)
                {
                    texture = Cvec3(0, g_hairyness);
                }
                else if (k == 1)
                {
                    texture = Cvec3(g_hairyness, 0);
                }
                else
                {
                    texture = Cvec3(0, 0);
                }

                verts.push_back(VertexPNX(position[0], position[1], position[2], normal[0], normal[1], normal[2], texture[0], texture[1]));
            }
        }
        g_bunnyShellGeometries[i]->upload(&verts[0], verts.size());
        verts.clear();
    }
    //g_shellNeedsUpdate = false;
}


// New function to update the simulation every frame
static void hairsSimulationUpdate() {
    Matrix4 worldToBunny = rigTFormToMatrix(getPathAccumRbt(g_world, g_bunnyNode));
    for (int i = 0; i < g_tipPos.size(); i++)
    {
        Cvec3 p = g_bunnyMesh.getVertex(i).getPosition();
        p = Cvec3(inv(worldToBunny) * Cvec4(p, 1));

        Cvec3 s = p + (g_bunnyMesh.getVertex(i).getNormal() * g_furHeight);
        s = Cvec3(inv(worldToBunny) * Cvec4(s, 1));

        Cvec3 g = g_gravity;
        Cvec3 t = Cvec3(inv(worldToBunny) * Cvec4(g_tipPos[i], 1));
        Cvec3 h = (s - t) * g_stiffness;
        Cvec3 f = g + h;

        Cvec3 temp = t + g_tipVelocity[i] * g_timeStep;
        Cvec3 tMinP = temp - p;
        temp = p + ((tMinP) / sqrt(dot(tMinP, tMinP))) * g_furHeight;
        g_tipPos[i] = Cvec3(worldToBunny * Cvec4(temp, 1));

        g_tipVelocity[i] = (g_tipVelocity[i] + (f * g_timeStep)) * g_damping;
    }

    g_shellNeedsUpdate = true;
    //updateShellGeometry();
    
}



static void initBunnyMeshes() {
    g_bunnyMesh.load("bunny.mesh");

    initNormals(g_bunnyMesh);
    
    vector<VertexPN> verts;
    verts.reserve(g_bunnyMesh.getNumFaces() * 6);

    for (int i = 0; i < g_bunnyMesh.getNumFaces(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            verts.push_back(getVertexPN(g_bunnyMesh, i, j));
        }
    }

    
    g_bunnyGeometry.reset(new SimpleGeometryPN(&verts[0], verts.size()));

    // Now allocate array of SimpleGeometryPNX to for shells, one per layer
    g_bunnyShellGeometries.resize(g_numShells);
    for (int i = 0; i < g_numShells; ++i) {
        g_bunnyShellGeometries[i].reset(new SimpleGeometryPNX());
    }
}








// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
    uniforms.put("uProjMatrix", projMatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY()
{
    if (g_windowWidth >= g_windowHeight)
        g_frustFovY = g_frustMinFov;
    else
    {
        const double RAD_PER_DEG = 0.5 * CS175_PI / 180;
        g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight /
                                g_windowWidth,
                            cos(g_frustMinFov * RAD_PER_DEG)) /
                      RAD_PER_DEG;
    }
}

static Matrix4 makeProjectionMatrix()
{
    return Matrix4::makeProjection(
        g_frustFovY, g_windowWidth / static_cast<double>(g_windowHeight),
        g_frustNear, g_frustFar);
}

enum ManipMode
{
    ARCBALL_ON_PICKED,
    ARCBALL_ON_SKY,
    EGO_MOTION
};

static ManipMode getManipMode()
{
    if (g_currentPickedRbtNode == g_activeEye)
    {
        if (g_activeEye == g_skyNode && g_activeCameraFrame == WORLD_SKY)
            return ARCBALL_ON_SKY;
        else
            return EGO_MOTION;
    }
    else
        return ARCBALL_ON_PICKED;
}

static bool shouldUseArcball()
{

    return !(g_currentPickedRbtNode == g_skyNode);
    //  return getManipMode() != EGO_MOTION && (!(g_activeEye != SKY &&
    //  g_currentPickedRbtNode == SKY));
}

// The translation part of the aux frame either comes from the current
// active object, or is the identity matrix when
static RigTForm getArcballRbt()
{
    switch (getManipMode())
    {
    case ARCBALL_ON_PICKED:
        return getPathAccumRbt(g_world, g_currentPickedRbtNode);
    case ARCBALL_ON_SKY:
        return RigTForm();
    case EGO_MOTION:
        return g_currentPickedRbtNode->getRbt();
    default:
        throw runtime_error("Invalid ManipMode");
    }
}

static void updateArcballScale()
{
    RigTForm arcballEye = inv(g_activeEye->getRbt()) * getArcballRbt();
    double depth = arcballEye.getTranslation()[2];
    if (depth > -CS175_EPS)
        g_arcballScale = 0.02;
    else
        g_arcballScale =
            getScreenToEyeScale(depth, g_frustFovY, g_windowHeight);
}

static void drawArcBall(Uniforms& uniforms)
{
    // switch to wire frame mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    RigTForm arcballEye = inv(getPathAccumRbt(g_world, g_activeEye)) * getArcballRbt();

    Matrix4 MVM = rigTFormToMatrix(arcballEye) *
                  Matrix4::makeScale(Cvec3(1, 1, 1) * g_arcballScale *
                                     g_arcballScreenRadius);
    
    // Use uniforms as opposed to curSS
    sendModelViewNormalMatrix(uniforms, MVM, normalMatrix(MVM));
    uniforms.put("uColor", Cvec3(.1, .82, .27));
    g_arcballMat->draw(*g_sphere, uniforms);
}

static void drawStuff(bool picking)
{
    if (g_shellNeedsUpdate) {
        updateShellGeometry();
    }
    
    // Declare an empty uniforms
    Uniforms uniforms;

    // if we are not translating, update arcball scale
    if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) ||
          (g_mouseLClickButton && !g_mouseRClickButton && g_spaceDown))) {
        updateArcballScale();
    }
        

    // build & send proj. matrix to vshader
    const Matrix4 projmat = makeProjectionMatrix();
    sendProjectionMatrix(uniforms, projmat);

    const RigTForm eyeRbt = getPathAccumRbt(g_world, g_activeEye);
    const RigTForm invEyeRbt = inv(eyeRbt);

    Cvec3 light1 = getPathAccumRbt(g_world, g_light1Node).getTranslation();
    Cvec3 light2 = getPathAccumRbt(g_world, g_light2Node).getTranslation();

    // transform to eye space and set to uLight uniform
    uniforms.put("uLight", Cvec3(invEyeRbt * Cvec4(light1, 1)));
    uniforms.put("uLight2", Cvec3(invEyeRbt * Cvec4(light2, 1)));

    if (!picking)
    {
        Drawer drawer(invEyeRbt, uniforms);
        g_world->accept(drawer);

        // draw arcball as part of asst3
        if (g_displayArcball && shouldUseArcball())
        {
            drawArcBall(uniforms);
        }
    }
    else
    {
        Picker picker(invEyeRbt, uniforms);
        g_overridingMaterial = g_pickingMat;
        g_world->accept(picker);
        g_overridingMaterial.reset();

        glFlush();
        // The OpenGL framebuffer uses pixel units, but it reads mouse coordinates
        // using point units. Most of the time these match, but on some hi-res
        // screens there can be a scaling factor.
        g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX * g_wScale,
                                                       g_mouseClickY * g_hScale);
        if (g_currentPickedRbtNode == g_groundNode || g_currentPickedRbtNode == 0)
        {
            g_currentPickedRbtNode = g_skyNode; // set to skyNode
            g_picking = false;
        }
    }
}

static void pick()
{
    // We need to set the clear color to black, for pick rendering.
    // so let's save the clear color
    GLdouble clearColor[4];
    glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // No more glUseProgram
    drawStuff(true); // no more curSS

    //Now set back the clear color
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

    checkGlErrors();
}

static void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawStuff(false);
    glfwSwapBuffers(g_window);
    checkGlErrors();
}

static void reshape(GLFWwindow *window, const int w, const int h)
{
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);
    glViewport(0, 0, width, height);

    g_windowWidth = w;
    g_windowHeight = h;
    cerr << "Size of window is now " << g_windowWidth << "x" << g_windowHeight << endl;
    g_arcballScreenRadius = max(1.0, min(h, w) * 0.1);
    updateFrustFovY();
}

static Cvec3 getArcballDirection(const Cvec2 &p, const double r)
{
    double n2 = norm2(p);
    if (n2 >= r * r)
        return normalize(Cvec3(p, 0));
    else
        return normalize(Cvec3(p, sqrt(r * r - n2)));
}

static RigTForm moveArcball(const Cvec2 &p0, const Cvec2 &p1)
{
    const Matrix4 projMatrix = makeProjectionMatrix();
    const RigTForm eyeInverse = inv(getPathAccumRbt(g_world, g_activeEye));
    const Cvec3 arcballCenter = getArcballRbt().getTranslation();
    cerr << "position" << arcballCenter[0] << endl;
    const Cvec3 arcballCenter_ec = Cvec3(eyeInverse * Cvec4(arcballCenter, 1));

    if (arcballCenter_ec[2] > -CS175_EPS)
        return RigTForm();

    Cvec2 ballScreenCenter =
        getScreenSpaceCoord(arcballCenter_ec, projMatrix, g_frustNear,
                            g_frustFovY, g_windowWidth, g_windowHeight);
    const Cvec3 v0 =
        getArcballDirection(p0 - ballScreenCenter, g_arcballScreenRadius);
    const Cvec3 v1 =
        getArcballDirection(p1 - ballScreenCenter, g_arcballScreenRadius);

    return RigTForm(Quat(0.0, v1[0], v1[1], v1[2]) *
                    Quat(0.0, -v0[0], -v0[1], -v0[2]));
}

static RigTForm doMtoOwrtA(const RigTForm &M, const RigTForm &O,
                           const RigTForm &A)
{
    return A * M * inv(A) * O;
}

static RigTForm getMRbt(const double dx, const double dy)
{
    RigTForm M;

    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown)
    {
        if (shouldUseArcball())
            M = moveArcball(Cvec2(g_mouseClickX, g_mouseClickY),
                            Cvec2(g_mouseClickX + dx, g_mouseClickY + dy));
        else
            M = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
    }
    else
    {
        double movementScale =
            getManipMode() == EGO_MOTION ? 0.02 : g_arcballScale;
        if (g_mouseRClickButton && !g_mouseLClickButton)
        {
            M = RigTForm(Cvec3(dx, dy, 0) * movementScale);
        }
        else if (g_mouseMClickButton ||
                 (g_mouseLClickButton && g_mouseRClickButton) ||
                 (g_mouseLClickButton && g_spaceDown))
        {
            M = RigTForm(Cvec3(0, 0, -dy) * movementScale);
        }
    }

    switch (getManipMode())
    {
    case ARCBALL_ON_PICKED:
        break;
    case ARCBALL_ON_SKY:
        M = inv(M);
        break;
    case EGO_MOTION:
        //    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown)
        //    // only invert rotation
        M = inv(M);
        break;
    }
    return M;
}

static RigTForm makeMixedFrame(const RigTForm &objRbt, const RigTForm &eyeRbt)
{
    return transFact(objRbt) * linFact(eyeRbt);
}

static void motion(GLFWwindow *window, double x, double y)
{
    if (!g_mouseClickDown)
        return;
    if (g_currentPickedRbtNode == g_skyNode && g_activeEye != g_skyNode)
        return; // we do not edit the sky when viewed from the objects

    const double dx = x - g_mouseClickX;
    const double dy = g_windowHeight - y - 1 - g_mouseClickY;

    const RigTForm M = getMRbt(dx, dy); // the "action" matrix

    // the matrix for the auxiliary frame (the w.r.t.)
    const RigTForm A =
        makeMixedFrame(getArcballRbt(), getPathAccumRbt(g_world, g_activeEye));

    RigTForm O = doMtoOwrtA(M, getArcballRbt(), A);

    if ((g_mouseLClickButton && !g_mouseRClickButton &&
         !g_spaceDown) // rotating
        && g_currentPickedRbtNode == g_skyNode)
    {
        RigTForm My = getMRbt(dx, 0);
        RigTForm Mx = getMRbt(0, dy);
        RigTForm B = makeMixedFrame(getArcballRbt(), RigTForm());
        O = doMtoOwrtA(Mx, g_currentPickedRbtNode->getRbt(), A);
        O = doMtoOwrtA(My, O, B);
    }
    else
    {
        RigTForm As = inv(getPathAccumRbt(g_world, g_currentPickedRbtNode, 1)) * A;
        O = doMtoOwrtA(M, g_currentPickedRbtNode->getRbt(), As);
    }

    g_currentPickedRbtNode->setRbt(O);

    g_mouseClickX += dx;
    g_mouseClickY += dy;
}

static void mouse(GLFWwindow *window, int button, int state, int mods)
{
    double x, y;
    glfwGetCursorPos(window, &x, &y);

    g_mouseClickX = x;
    g_mouseClickY =
        g_windowHeight - y - 1; // conversion from GLUT window-coordinate-system
                                // to OpenGL window-coordinate-system

    g_mouseLClickButton |= (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS);
    g_mouseRClickButton |= (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS);
    g_mouseMClickButton |= (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS);

    g_mouseLClickButton &= !(button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_RELEASE);
    g_mouseRClickButton &= !(button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_RELEASE);
    g_mouseMClickButton &= !(button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_RELEASE);

    g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

    if (g_picking && g_mouseLClickButton && !g_mouseRClickButton)
    {
        cout << "1"
             << "\n";
        pick();
        cout << "currently picked: " << g_currentPickedRbtNode << "\n";
    }
}

static void createKeyFrame()
{
    vector<RigTForm> frame;
    for (vector<shared_ptr<SgRbtNode> >::iterator iter = g_tempFrame.begin(); iter != g_tempFrame.end(); ++iter)
    {
        frame.push_back((**iter).getRbt());
    }
    if (g_keyFrames.size() > 0)
    {
        g_keyFrames.insert(++g_currentFrame, frame); // latest frame
        --g_currentFrame;
    }
    else
    {
        g_keyFrames.insert(g_keyFrames.begin(), frame);
        g_currentFrame = g_keyFrames.begin();
    }
}

// interpolate frames
static vector<RigTForm> linearInterpolation(vector<RigTForm> frame1, vector<RigTForm> frame2, float t)
{
    // declare a key
    vector<RigTForm> frames;
    vector<RigTForm>::iterator iter2 = frame2.begin();
    ;
    // iterate through first frame
    for (vector<RigTForm>::iterator iter1 = frame1.begin(); iter1 != frame1.end(); ++iter1)
    {
        float alpha = t - floor(t);

        // find the rotation
        Quat q0 = iter1->getRotation();
        Quat q1 = iter2->getRotation();
        Quat q = q1 * inv(q0);
        // absolute value
        if (q[0] < 0)
            q = q * (-1);
        // find the translation
        Cvec3 t0 = iter1->getTranslation();
        Cvec3 t1 = iter2->getTranslation();
        Cvec3 trans = t0 * (1 - alpha) + t1 * alpha;

        // theta
        double theta = atan2(double(sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])), q[0]);

        // quat
        if (theta > CS175_EPS)
        {
            double theta_new = theta * alpha; // new theta value
            double diff = (sin(theta_new) / sin(theta));
            q = Quat(cos(theta_new), q[1] * diff, q[2] * diff, q[3] * diff) * q0;
        }
        else
        {
            q = Quat();
        }
        frames.push_back(RigTForm(trans, q));
        ++iter2;
    }

    return frames;
}

static vector<RigTForm> CRInterpolation(const double alpha, vector<RigTForm> res, vector<RigTForm> frame0, vector<RigTForm> frame1, vector<RigTForm> frame2, vector<RigTForm> frame3)
{

    vector<RigTForm>::iterator iter0 = frame0.begin();
    vector<RigTForm>::iterator iter1 = frame1.begin();
    vector<RigTForm>::iterator iter2 = frame2.begin();
    vector<RigTForm>::iterator iter3 = frame3.begin();

    for (vector<RigTForm>::iterator i = res.begin(); i != res.end(); ++i)
    {
        Cvec3 trans0 = iter0->getTranslation();
        Cvec3 trans1 = iter1->getTranslation();
        Cvec3 trans2 = iter2->getTranslation();
        Cvec3 trans3 = iter3->getTranslation();
        Cvec3 trans;

        for (int j = 0; j < 3; j++)
        {
            double d = ((1.0 / 6.0) * (trans2[j] - trans0[j])) + trans1[j];
            double e = ((-1.0 / 6.0) * (trans3[j] - trans1[j])) + trans2[j];
            double f = ((1 - alpha) * trans1[j]) + (alpha * d);
            double g = ((1 - alpha) * d) + (alpha * e);
            double h = ((1 - alpha) * e) + (alpha * trans2[j]);
            double m = ((1 - alpha) * f) + (alpha * g);
            double n = ((1 - alpha) * g) + (alpha * h);
            trans[j] = ((1 - alpha) * m) + (alpha * n);
        }

        (*i).setTranslation(trans);
        ++iter0;
        ++iter1;
        ++iter2;
        ++iter3;
    }
    return res;
}

static Quat quatPreprocess(Quat quat0, Quat quat1, const double alpha)
{
    Quat quat0_inv = inv(quat0);
    Quat q = quat1 * quat0_inv;

    if (q[0] < 0)
    {
        q = q * (-1);
    }

    double theta0 = atan2(double(sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])), q[0]);
    double theta1 = theta0 * alpha;

    if (theta0 > CS175_EPS)
    {
        double temp = (sin(theta1) / sin(theta0));
        q = Quat(cos(theta1), q[1] * temp, q[2] * temp, q[3] * temp);
        q = q * quat0;
    }
    else
    {
        q = Quat();
    }

    return q;
}

static vector<RigTForm> quatInterpolation(const double alpha, vector<RigTForm> res, vector<RigTForm> frame0, vector<RigTForm> frame1, vector<RigTForm> frame2, vector<RigTForm> frame3)
{

    vector<RigTForm>::iterator iter0 = frame0.begin();
    vector<RigTForm>::iterator iter1 = frame1.begin();
    vector<RigTForm>::iterator iter2 = frame2.begin();
    vector<RigTForm>::iterator iter3 = frame3.begin();

    for (vector<RigTForm>::iterator i = res.begin(); i != res.end(); ++i)
    {
        Quat quat0 = iter0->getRotation();
        Quat quat1 = iter1->getRotation();
        Quat quat2 = iter2->getRotation();
        Quat quat3 = iter3->getRotation();

        Quat q[2] = {(quat2 * inv(quat0)), (quat3 * inv(quat1))};
        for (int j = 0; j < 2; j++)
        {
            if (q[j][0] < 0)
            {
                q[j] = q[j] * (-1);
            }

            double theta0 = atan2(double(sqrtf(q[j][1] * q[j][1] + q[j][2] * q[j][2] + q[j][3] * q[j][3])), q[j][0]);
            double theta1;
            if (j == 0)
            {
                theta1 = theta0 * (1.0 / 6.0);
            }
            else
            {
                theta1 = theta0 * (-1.0 / 6.0);
            }

            if (theta0 > CS175_EPS)
            {
                double temp = (sin(theta1) / sin(theta0));
                q[j] = Quat(cos(theta1), q[j][1] * temp, q[j][2] * temp, q[j][3] * temp);
            }
            else
            {
                q[j] = Quat();
            }
        }

        Quat d = q[0] * quat1;
        Quat e = q[1] * quat2;

        Quat f = quatPreprocess(quat1, d, alpha);
        Quat g = quatPreprocess(d, e, alpha);
        Quat h = quatPreprocess(e, quat2, alpha);

        Quat m = quatPreprocess(f, g, alpha);
        Quat n = quatPreprocess(g, h, alpha);

        (*i).setRotation(quatPreprocess(m, n, alpha));

        ++iter0;
        ++iter1;
        ++iter2;
        ++iter3;
    }
    return res;
}

// Given t in the range [0, n], perform linearInterpolation and draw the scene
// for the particular t. Returns true if we are at the end of the animation
// sequence, or false otherwise.
bool interpolate(float t)
{ // check if reach end
    if (floor(t) >= g_keyFrames.size() - 3)
    {
        return true;
    }

    float alpha = t - floor(t);

    list<vector<RigTForm> >::iterator iter = g_keyFrames.begin();
    advance(iter, floor(t));

    vector<RigTForm> frame0 = *iter;
    iter++;
    vector<RigTForm> frame1 = *iter;
    iter++;
    vector<RigTForm> frame2 = *iter;
    iter++;
    vector<RigTForm> frame3 = *iter;

    vector<RigTForm> res;
    for (int i = 0; i < int(frame0.size()); ++i)
    {
        res.push_back(RigTForm());
    }
    res = CRInterpolation(alpha, res, frame0, frame1, frame2, frame3);
    res = quatInterpolation(alpha, res, frame0, frame1, frame2, frame3);

    // scene graph
    vector<RigTForm>::iterator iter2 = res.begin();
    for (vector<shared_ptr<SgRbtNode> >::iterator iter1 = g_tempFrame.begin(); iter1 != g_tempFrame.end(); ++iter1)
    {
        (**iter1).setRbt(*iter2);
        ++iter2;
    }

    return false;
}

// Call every frame to advance the animation
void animationUpdate()
{
    if (g_playingAnimation)
    {
        float t = (float)g_animateTime / g_msBetweenKeyFrames;
        bool endReached = interpolate(t);

        if (!endReached)
            g_animateTime += 1000. / g_framesPerSecond;
        else
        {
            // finish and clean up
            g_playingAnimation = false;
            g_animateTime = 0;
            // to do
        }
    }
}

static void fread()
{
    ifstream myfile("test.txt");

    int count = -2;
    int num_keyframes = 0;
    int num_rbts = 0;
    int rigt_count = 0;
    double d;
    double buffer[10];
    vector<RigTForm> vbuffer;

    while (!myfile.eof())
    {
        myfile >> d;

        if (count == -2)
            num_keyframes = d;
        else if (count == -1)
            num_rbts = d;
        else
        {
            if (count < 10)
                buffer[count] = d;
            if (count == 9)
            {
                vbuffer.push_back(RigTForm(Cvec3(buffer[4], buffer[5], buffer[6]), Quat(buffer[0], buffer[1], buffer[2], buffer[3])));
                rigt_count++;
                count = -1;
            }
        }
        if (rigt_count == num_rbts)
        {
            if (g_keyFrames.size() > 0)
            {
                g_keyFrames.insert(++g_currentFrame, vbuffer);
                --g_currentFrame;
            }
            else
            {
                g_keyFrames.insert(g_keyFrames.begin(), vbuffer);
                g_currentFrame = g_keyFrames.begin();
            }
            vbuffer.clear();
            rigt_count = 0;
        }
        ++count;
    }
    g_currentFrame = g_keyFrames.end();
    --g_currentFrame;
    cout << "File has been successfully read" << endl;
}

static void fwrite()
{
    ofstream outf("test.txt");

    // fail to open
    if (!outf)
    {
        cerr << "test.txt does not exist" << endl;
        exit(1);
    }

    // write frames and RigTForm per frame
    outf << g_keyFrames.size() << " " << g_keyFrames.begin()->size() << "\n";
    for (list<vector<RigTForm> >::iterator iter1 = g_keyFrames.begin(); iter1 != g_keyFrames.end(); ++iter1)
    {
        for (vector<RigTForm>::iterator iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2)
        {
            Quat iter_Ro = iter2->getRotation();
            Cvec3 iter_Trans = iter2->getTranslation();
            outf << iter_Ro[0] << " " << iter_Ro[1] << " " << iter_Ro[2] << " " << iter_Ro[3] << " " << iter_Trans[0] << " " << iter_Trans[1] << " " << iter_Trans[2] << "\n";
        }
    }
}

static void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            exit(0);
        case GLFW_KEY_H:
            cout << " ============== H E L P ==============\n\n"
                 << "h\t\thelp menu\n"
                 << "s\t\tsave screenshot\n"
                 << "f\t\tToggle flat shading on/off.\n"
                 << "o\t\tCycle object to edit\n"
                 << "v\t\tCycle view\n"
                 << "drag left mouse to rotate\n"
                 << endl;
            break;
        case GLFW_KEY_S:
            glFlush();
            writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
            break;
        case GLFW_KEY_V:
            g_activeEyeIndex = (g_activeEyeIndex + 1) % 4;
            if (g_activeEyeIndex == 0)
                g_activeEye = g_skyNode;
            else if (g_activeEyeIndex == 1)
                g_activeEye = g_groundNode;
            else if (g_activeEyeIndex == 2)
                g_activeEye = g_robot1Node;
            else
                g_activeEye = g_robot2Node;
            cerr << "Active eye is " << g_objNames[g_activeEyeIndex] << endl;
            break;
        case GLFW_KEY_M:
            g_activeCameraFrame = SkyMode((g_activeCameraFrame + 1) % 2);
            cerr << "Editing sky eye w.r.t. "
                 << (g_activeCameraFrame == WORLD_SKY ? "world-sky frame\n"
                                                      : "sky-sky frame\n")
                 << endl;
            break;
        case GLFW_KEY_P:
            g_picking = true;
            cerr << "Prepare to pick" << endl;
            break;
        case GLFW_KEY_SPACE:
            g_spaceDown = true;
            break;
        case GLFW_KEY_C:
            if (g_keyFrames.size() > 0)
            {
                vector<RigTForm>::iterator iter1 = (*g_currentFrame).begin();
                vector<shared_ptr<SgRbtNode> >::iterator iter2;
                for (iter2 = g_tempFrame.begin(); iter2 != g_tempFrame.end(); ++iter2)
                {
                    (**iter2).setRbt(*iter1);
                    ++iter1;
                }
                cout << "Copy current key frame RBT data to the scene graph" << endl;
            }
            else
            {
                cout << "No key frame defined" << endl;
            }
            break;
        case GLFW_KEY_U:
            if (g_keyFrames.size() > 0)
            {
                vector<RigTForm>::iterator iter1 = (*g_currentFrame).begin();
                vector<shared_ptr<SgRbtNode> >::iterator iter2;
                for (iter2 = g_tempFrame.begin(); iter2 != g_tempFrame.end(); ++iter2)
                {
                    *iter1 = (**iter2).getRbt();
                    ++iter1;
                }
                cout << "Copy the scene graph to the current frame" << endl;
            }
            else
            {
                createKeyFrame();
            }
            break;
        case GLFW_KEY_PERIOD:
            if (!(mods & GLFW_MOD_SHIFT))
                break;
            if (g_currentFrame != --g_keyFrames.end())
            {
                ++g_currentFrame;
                vector<shared_ptr<SgRbtNode> >::iterator iter1;
                vector<RigTForm>::iterator iter2 = (*g_currentFrame).begin();
                for (iter1 = g_tempFrame.begin(); iter1 != g_tempFrame.end(); ++iter1)
                {
                    (**iter1).setRbt(*iter2);
                    ++iter2;
                }
                cout << "Advance to next frame" << endl;
            }
            else
            {
                cout << "Cannot advance to next frame" << endl;
            }
            break;
        case GLFW_KEY_COMMA:
            if (!(mods & GLFW_MOD_SHIFT))
                break;
            if (g_currentFrame != g_keyFrames.begin())
            {
                --g_currentFrame;
                vector<shared_ptr<SgRbtNode> >::iterator iter1;
                vector<RigTForm>::iterator iter2 = (*g_currentFrame).begin();
                for (iter1 = g_tempFrame.begin(); iter1 != g_tempFrame.end(); ++iter1)
                {
                    (**iter1).setRbt(*iter2);
                    ++iter2;
                }
                cout << "Retreat to previous frame" << endl;
            }
            else
            {
                cout << "Cannot retreat to previous frame" << endl;
            }
            break;
        case GLFW_KEY_D:
            if (g_keyFrames.size() > 0)
            {
                if (g_keyFrames.size() == 1)
                {
                    g_keyFrames.erase(g_currentFrame);
                    g_currentFrame = g_keyFrames.end();
                    cout << "Frame list is now EMPTY" << endl;
                }
                else
                {
                    if (g_currentFrame != g_keyFrames.begin())
                    {
                        list<vector<RigTForm> >::iterator iter = g_currentFrame;
                        --iter;
                        g_keyFrames.erase(g_currentFrame);
                        g_currentFrame = g_keyFrames.begin();
                    }
                    else
                    {
                        g_keyFrames.erase(g_currentFrame);
                        g_currentFrame = g_keyFrames.begin();
                    }
                    vector<RigTForm>::iterator iter1 = (*g_currentFrame).begin();
                    vector<shared_ptr<SgRbtNode> >::iterator iter2;
                    for (iter2 = g_tempFrame.begin(); iter2 != g_tempFrame.end(); ++iter2)
                    {
                        (**iter2).setRbt(*iter1);
                        ++iter1;
                    }
                    cout << "Delete the current key frame" << endl;
                }
            }
            break;
        case GLFW_KEY_N:
            createKeyFrame();
            cout << "Create a new frame" << endl;
            break;
        case GLFW_KEY_I:
            fread();
            cout << "Input key frames from input file" << endl;
            break;
        case GLFW_KEY_W:
            fwrite();
            cout << "Output key frames to output file" << endl;
            break;
        case GLFW_KEY_Y:
            if (g_keyFrames.size() < 4)
            {
                cout << "Don't have enough frames to play the animation " << endl;
            }
            else
            {
                g_playingAnimation = true;
                animationUpdate();
            }
            break;
        case GLFW_KEY_EQUAL:
            if (!(mods & GLFW_MOD_SHIFT))
                break;
            if (g_msBetweenKeyFrames > 100)
                g_msBetweenKeyFrames -= 100; // ??? ccomplished by changing g msBetweenKeyFrames
            cout << "Make the animation go faster, " << g_msBetweenKeyFrames << "ms between frames" << endl;
            break;
        case GLFW_KEY_MINUS:
            if (!(mods & GLFW_MOD_SHIFT))
                break;
            g_msBetweenKeyFrames += 100; // ??? ccomplished by changing g msBetweenKeyFrames
            cout << "Make the animation go slower, " << g_msBetweenKeyFrames << "ms between frames" << endl;
            break;
        case GLFW_KEY_RIGHT:
            g_furHeight *= 1.05;
            cerr << "fur height = " << g_furHeight << std::endl;
            break;
        case GLFW_KEY_LEFT:
            g_furHeight /= 1.05;
            std::cerr << "fur height = " << g_furHeight << std::endl;
            break;
        case GLFW_KEY_UP:
            g_hairyness *= 1.05;
            cerr << "hairyness = " << g_hairyness << std::endl;
            break;
        case GLFW_KEY_DOWN:
            g_hairyness /= 1.05;
            cerr << "hairyness = " << g_hairyness << std::endl;
            break;
        }
    }
    else
    {
        switch (key)
        {
        case GLFW_KEY_SPACE:
            g_spaceDown = false;
            break;
        }
    }
}

void error_callback(int error, const char *description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void initGlfwState()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    g_window = glfwCreateWindow(g_windowWidth, g_windowHeight,
                                "Assignment 9", NULL, NULL);
    if (!g_window)
    {
        fprintf(stderr, "Failed to create GLFW window or OpenGL context\n");
        exit(1);
    }
    glfwMakeContextCurrent(g_window);
    glewInit();

    glfwSwapInterval(1);

    glfwSetErrorCallback(error_callback);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetKeyCallback(g_window, keyboard);

    int screen_width, screen_height;
    glfwGetWindowSize(g_window, &screen_width, &screen_height);
    int pixel_width, pixel_height;
    glfwGetFramebufferSize(g_window, &pixel_width, &pixel_height);

    cout << screen_width << " " << screen_height << endl;
    cout << pixel_width << " " << pixel_width << endl;

    g_wScale = pixel_width / screen_width;
    g_hScale = pixel_height / screen_height;
}

static void initGLState()
{
    glClearColor(128. / 255., 200. / 255., 255. / 255., 0.);
    glClearDepth(0.);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glReadBuffer(GL_BACK);
    if (!g_Gl2Compatible)
        glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initMaterials() {
    // Create some prototype materials
    Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
    Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

    // copy diffuse prototype and set red color
    g_redDiffuseMat.reset(new Material(diffuse));
    g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

    // copy diffuse prototype and set blue color
    g_blueDiffuseMat.reset(new Material(diffuse));
    g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

    // normal mapping material
    g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
    g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
    g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

    // copy solid prototype, and set to wireframed rendering
    g_arcballMat.reset(new Material(solid));
    g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
    g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // copy solid prototype, and set to color white
    g_lightMat.reset(new Material(solid));
    g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

    // pick shader
    g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));
    
    // bunny material
    g_bunnyMat.reset(new Material("./shaders/basic-gl3.vshader",
                                  "./shaders/bunny-gl3.fshader"));
    g_bunnyMat->getUniforms().put("uColorAmbient", Cvec3f(0.45f, 0.3f, 0.3f))
                             .put("uColorDiffuse", Cvec3f(0.2f, 0.2f, 0.2f));

    // bunny shell materials;
    // common shell texture:
    shared_ptr<ImageTexture> shellTexture(new ImageTexture("shell.ppm", false));

    // enable repeating of texture coordinates
    shellTexture->bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // eachy layer of the shell uses a different material, though the materials
    // will share the same shader files and some common uniforms. hence we
    // create a prototype here, and will copy from the prototype later
    Material bunnyShellMatPrototype("./shaders/bunny-shell-gl3.vshader",
                                    "./shaders/bunny-shell-gl3.fshader");
    bunnyShellMatPrototype.getUniforms().put("uTexShell", shellTexture);
    bunnyShellMatPrototype.getRenderStates()
        .blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) // set blending mode
        .enable(GL_BLEND)                                // enable blending
        .disable(GL_CULL_FACE);                          // disable culling

    // allocate array of materials
    g_bunnyShellMats.resize(g_numShells);
    for (int i = 0; i < g_numShells; ++i) {
        // copy prototype
        g_bunnyShellMats[i].reset(new Material(bunnyShellMatPrototype));
        // but set a different exponent for blending transparency
        g_bunnyShellMats[i]->getUniforms().put(
            "uAlphaExponent", 2.f + 5.f * float(i + 1) / g_numShells);
    }
};

static void initGeometry()
{
    initGround();
    initCubes();
    initSphere();
    initBunnyMeshes();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material)
{

    const float ARM_LEN = 0.7,
                ARM_THICK = 0.25,
                TORSO_LEN = 1.5,
                TORSO_THICK = 0.25,
                TORSO_WIDTH = 1,
                HEAD_RADIUS = 0.35;
    const int NUM_JOINTS = 10,
              NUM_SHAPES = 10;

    struct JointDesc
    {
        int parent;
        float x, y, z;
    };

    JointDesc jointDesc[NUM_JOINTS] = {
        {-1},                                    // torso
        {0, TORSO_WIDTH / 2, TORSO_LEN / 2, 0},  // upper right arm
        {1, ARM_LEN, 0, 0},                      // lower right arm
        {0, -TORSO_WIDTH / 2, TORSO_LEN / 2, 0}, // upper left arm
        {3, -ARM_LEN, 0, 0},                     // lower left arm
        {0, ARM_LEN / 2, -TORSO_LEN / 2, 0},     // upper right leg
        {5, 0, -ARM_LEN, 0},                     // lower right leg
        {0, -ARM_LEN / 2, -TORSO_LEN / 2, 0},    // upper left leg
        {7, 0, -ARM_LEN, 0},                     // lower left leg
        {0, 0, TORSO_WIDTH / 2, 0}               // head
    };

    struct ShapeDesc
    {
        int parentJointId;
        float x, y, z, sx, sy, sz;
        shared_ptr<Geometry> geometry;
    };

    ShapeDesc shapeDesc[NUM_SHAPES] = {
        {0, 0, 0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube},                              // torso
        {1, ARM_LEN / 2, 0, 0, ARM_LEN / 2, ARM_THICK / 2, ARM_THICK / 2, g_sphere},            // upper right arm
        {2, ARM_LEN / 2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube},                          // lower right arm
        {3, -ARM_LEN / 2, 0, 0, -ARM_LEN / 2, ARM_THICK / 2, ARM_THICK / 2, g_sphere},          // upper left arm
        {4, -ARM_LEN / 2, 0, 0, -ARM_LEN, ARM_THICK, ARM_THICK, g_cube},                        // lower left arm
        {5, 0, -ARM_LEN / 2, 0, ARM_THICK / 2, -ARM_LEN / 2, ARM_THICK / 2, g_sphere},          // upper right leg
        {6, 0, -ARM_LEN / 2, 0, ARM_THICK, -ARM_LEN, ARM_THICK, g_cube},                        // lower right leg
        {7, 0, -ARM_LEN / 2, 0, ARM_THICK / 2, -ARM_LEN / 2, ARM_THICK / 2, g_sphere},          // upper left leg
        {8, 0, -ARM_LEN / 2, 0, ARM_THICK, -ARM_LEN, ARM_THICK, g_cube},                        // lower left leg
        {9, 0, TORSO_WIDTH / 2, 0, TORSO_WIDTH / 2, TORSO_WIDTH / 2, TORSO_THICK / 2, g_sphere} // head
    };

    shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

    for (int i = 0; i < NUM_JOINTS; ++i)
    {
        if (jointDesc[i].parent == -1)
            jointNodes[i] = base;
        else
        {
            jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
            jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
        }
    }
    for (int i = 0; i < NUM_SHAPES; ++i) {
        shared_ptr<SgGeometryShapeNode> shape(
            new MyShapeNode(shapeDesc[i].geometry,
                material, // USE MATERIAL as opposed to color
                Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                Cvec3(0, 0, 0),
                Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
    }
}




static void initScene() {
    g_world.reset(new SgRootNode());

    g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 8.0))));

    g_groundNode.reset(new SgRbtNode(RigTForm(Cvec3(0, g_groundY, 0))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_bumpFloorMat)));

    g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-8, 1, 0))));
    g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(8, 1, 0))));

    constructRobot(g_robot1Node, g_redDiffuseMat);
    constructRobot(g_robot2Node, g_blueDiffuseMat);

    g_light1Node.reset(new SgRbtNode(RigTForm(Cvec3(4.0, 3.0, 5.0))));
    g_light2Node.reset(new SgRbtNode(RigTForm(Cvec3(-4, 1.0, -4.0))));
    g_light1Node->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_sphere, g_lightMat, Cvec3(0), Cvec3(0), Cvec3(0.5))));

    g_light2Node->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_sphere, g_lightMat, Cvec3(0), Cvec3(0), Cvec3(0.5))));

    g_bunnyNode.reset(new SgRbtNode());

    g_bunnyNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_bunnyGeometry, g_bunnyMat)));

    for (int i = 0; i < g_numShells; ++i) {
        g_bunnyNode->addChild(shared_ptr<MyShapeNode>(
            new MyShapeNode(g_bunnyShellGeometries[i], g_bunnyShellMats[i])));
    }



    g_world->addChild(g_skyNode);
    g_world->addChild(g_groundNode);
    g_world->addChild(g_robot1Node);
    g_world->addChild(g_robot2Node);
    g_world->addChild(g_light1Node);
    g_world->addChild(g_light2Node);
    g_world->addChild(g_bunnyNode);

    g_currentPickedRbtNode = g_skyNode;
    g_activeEye = g_skyNode;
}

void glfwLoop()
{
    g_lastFrameClock = glfwGetTime();

    while (!glfwWindowShouldClose(g_window))
    {
        double thisTime = glfwGetTime();
        if (thisTime - g_lastFrameClock >= 1. / g_framesPerSecond)
        {

            animationUpdate();
            hairsSimulationUpdate();
            //updateShellGeometry();

            display();
            g_lastFrameClock = thisTime;
        }

        glfwPollEvents();
    }
}


// New function to initialize the dynamics simulation
static void initSimulation() {
    g_tipPos.resize(g_bunnyMesh.getNumVertices(), Cvec3(0));
    g_tipVelocity = g_tipPos;

    Matrix4 worldToBunny = rigTFormToMatrix(getPathAccumRbt(g_world, g_bunnyNode));

    for (int i = 0; i < g_bunnyMesh.getNumVertices(); ++i)
    {
        Mesh::Vertex v = g_bunnyMesh.getVertex(i);
        Cvec3 s = v.getPosition();
        s += (v.getNormal() * g_furHeight);
        g_tipPos[i] = s;
    }

    g_shellNeedsUpdate = true;
    updateShellGeometry();

    hairsSimulationUpdate();

}

int main(int argc, char *argv[])
{
    try
    {
        initGlfwState();

        // on Mac, we shouldn't use GLEW.
#ifndef __MAC__
        glewInit(); // load the OpenGL extensions

        if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
            throw runtime_error("Error: card/driver does not support OpenGL "
                                "Shading Language v1.3");
        else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
            throw runtime_error("Error: card/driver does not support OpenGL "
                                "Shading Language v1.0");
#endif

        cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0"
                                 : "Will use OpenGL 3.x / GLSL 1.5")
             << endl;

        initGLState();
        initMaterials();
        initGeometry();
        initScene();
        initSimulation();

        dumpSgRbtNodes(g_world, g_tempFrame);

        glfwLoop();

        return 0;
    }
    catch (const runtime_error &e)
    {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
