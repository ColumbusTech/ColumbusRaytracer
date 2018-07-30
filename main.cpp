#include "Raytracer.h"

int main(int argc, char** argv)
{
	Raytracer RT;
	RT.BackgroundColor = {0, 0, 0, 1};
	RT.ClearImage();
	RT.RenderScene();
	RT.SaveImage();

	return 0;
}




