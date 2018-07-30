#pragma once

#include <cstdint>
#include "Math.h"

class Raytracer
{
private:
	static constexpr uint32_t Width = 500;
	static constexpr uint32_t Height = 500;
	vec4 Image[Width][Height];
public:
	vec4 BackgroundColor;
	uint32_t RaysPerPixel = 4;

	float Aspect = (float)Width / (float)Height;
	float FOV = 60;
public:
	Raytracer();

	void ClearImage();

	void RenderLine(ivec2 A, ivec2 B, vec4 Col);

	void RenderScene();
	void SaveImage();

	~Raytracer();
};










