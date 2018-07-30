#include "Raytracer.h"
#include <cstdlib>
#include <cstdio>
#include <algorithm>

Raytracer::Raytracer() {}

void Raytracer::ClearImage()
{
	for (uint32_t X = 0; X < Width; X++)
	{
		for (uint32_t Y = 0; Y < Height; Y++)
		{
			Image[X][Y] = BackgroundColor;
		}
	}
}

void Raytracer::RenderLine(ivec2 A, ivec2 B, vec4 Col)
{
	#undef min

	float KoefWidth = 1.0f / Width;
	float KoefHeight = 1.0f / Height;

	for (float T = 0.0f; T < 1.0f; T += Math::Min(KoefWidth, KoefHeight))
	{
		uint32_t X = A.X * (1.0f - T) + B.X * T;
		uint32_t Y = A.Y * (1.0f - T) + B.Y * T;
		Image[X][Y] = Col;
	}
}

void Raytracer::RenderScene()
{
	Sphere Primitive({0, 0, -5}, {1, 0, 0, 1}, 1);
	PointLight Light({-2, 2, 0}, {1, 1, 1});

	int HalfWidth = Width / 2;
	int HalfHeight = Height / 2;
	float WidthStep = 1.0f / (HalfWidth);
	float HeightStep = 1.0f / (HalfHeight);
	float FOVStep = 1.0f / (180.0f / FOV);
	float AspectStep = 0.33f * Aspect;

	vec3 Normal;
	vec3 Hitpoint;

	for (int X = 0; X < Width; X++)
	{
		for (int Y = 0; Y < Height; Y++)
		{
			float RayDirectionX = (float)(X - HalfWidth) * WidthStep * FOVStep;
			float RayDirectionY = (float)(Y - HalfHeight) * HeightStep * AspectStep;
			vec3 RayDirection(RayDirectionX, RayDirectionY, -1);

			Ray R({0}, RayDirection);

			if (Primitive.Intersect(R, Normal, Hitpoint))
			{
				Image[X][Y] = vec4(Primitive.LightFunc(Light, Normal, Hitpoint, {0.2}, {0, 0, 0}), 1.0);
			}
		}
	}
}

void Raytracer::SaveImage()
{
	FILE* File = fopen("a.ppm", "w");
	fprintf(File, "P3\n%i %i\n%i\n", Width, Height, 255);

	for (uint32_t X = 0; X < Width; X++)
	{
		for (uint32_t Y = 0; Y < Height; Y++)
		{
			fprintf(File, "%i %i %i ", uint8_t(Image[X][Y].X * 255), uint8_t(Image[X][Y].Y * 255), uint8_t(Image[X][Y].Z * 255));
		}
	}

	fclose(File);
}

Raytracer::~Raytracer() {}











