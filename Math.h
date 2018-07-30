#include <cmath>
#include <cstdint>
#include <algorithm>

template <typename Type>
struct vec2_t;

template <typename Type>
struct vec3_t;

template <typename Type>
struct vec4_t;

template <typename Type>
struct mat4_t;

typedef vec2_t<float> vec2;
typedef vec2_t<double> dvec2;
typedef vec2_t<int> ivec2;
typedef vec2_t<bool> bvec2;

typedef vec3_t<float> vec3;
typedef vec3_t<double> dvec3;
typedef vec3_t<int> ivec3;
typedef vec3_t<bool> bvec3;

typedef vec4_t<float> vec4;
typedef vec4_t<double> dvec4;
typedef vec4_t<int> ivec4;
typedef vec4_t<bool> bvec4;

typedef mat4_t<float> mat4;
typedef mat4_t<double> dmat4;
typedef mat4_t<int> imat4;
typedef mat4_t<bool> bmat4;

struct Math
{
	template <typename Type>
	static constexpr Type Min(const Type& A, const Type& B) noexcept
	{
		return A < B ? A : B;
	}

	template <typename Type>
	static constexpr Type Max(const Type& A, const Type& B)  noexcept
	{
		return A > B ? A : B;
	}

	template <typename Type>
	static constexpr Type Clamp(const Type& MinValue, const Type& MaxValue, const Type& A) noexcept
	{
		return Min(Max(A, MinValue), MaxValue);
	}

	template <typename Type>
	static Type Mix(const Type& A, const Type& B, const double& Coef)
	{
		return A * (1.0 - Coef) + B * Coef; 
	}
};

template <typename Type>
struct vec2_t
{
	Type X;
	Type Y;
	Type Z;

	vec2_t() {}
	vec2_t(Type A) : X(A), Y(A) {}
	vec2_t(Type InX, Type InY) : X(InX), Y(InY) {}
	vec2_t(const vec2_t& Base) : X(Base.X), Y(Base.Y) {}
	vec2_t(vec2_t&& Base) :
		X(std::move(Base.X)),
		Y(std::move(Base.Y)) {}

	vec2_t& operator=(const vec2_t& Other)
	{
		X = Other.X;
		Y = Other.Y;
		return *this;
	}

	vec2_t& operator=(vec2_t&& Other)
	{
		X = std::move(Other.X);
		Y = std::move(Other.Y);
		return *this;
	}

	vec2_t operator+(const vec2_t& Other) const
	{
		return vec2_t(X + Other.X, Y + Other.Y);
	}

	vec2_t operator-() const
	{
		return vec2_t(-X, -Y);
	}

	vec2_t operator-(const vec2_t& Other) const
	{
		return vec2_t(X - Other.X, Y - Other.Y);
	}

	vec2_t operator*(const Type& Scalar) const
	{
		return vec2_t(X * Scalar, Y * Scalar);
	}

	friend vec2_t operator*(const Type& Scalar, const vec2_t& Other)
	{
		return Other * Scalar;
	}

	vec2_t operator*(const vec2_t& Other) const
	{
		return vec2_t(X * Other.X, Y * Other.Y);
	}

	vec2_t operator/(const Type& Scalar) const
	{
		const Type Factor = 1.0 / Scalar;
		return vec2_t(X * Factor, Y * Factor);
	}

	vec2_t operator/(const vec2_t& Other) const
	{
		return vec2_t(X / Other.X, Y / Other.Y);
	}

	vec2_t& operator+=(const vec2_t& Other)
	{
		return *this = *this + Other;
	}

	vec2_t& operator-=(const vec2_t& Other)
	{
		return *this = *this - Other;
	}

	vec2_t& operator*=(const Type& Scalar)
	{
		return *this = *this * Scalar;
	}

	vec2_t& operator*=(const vec2_t& Other)
	{
		return *this = *this * Other;
	}

	vec2_t& operator/=(const Type& Scalar)
	{
		return *this = *this / Scalar;
	}

	vec2_t& operator/=(const vec2_t& Other)
	{
		return *this = *this / Other;
	}

	vec2_t Clamped(const Type& Min, const Type& Max) const
	{
		return vec2_t(Math::Clamp(Min, Max, X), Math::Clamp(Min, Max, Y));
	}

	vec2_t Clamped(const vec2_t& Min, const vec2_t& Max) const
	{
		return vec2_t(Math::Clamp(Min.X, Max.X, X), Math::Clamp(Min.Y, Max.Y, Y));
	}

	vec2_t& Clamp(const Type& Min, const Type& Max)
	{
		return *this = Clamped(Min, Max);
	}

	vec2_t& Clamp(const vec2_t& Min, const vec2_t& Max)
	{
		return *this = Clamped(Min, Max);
	}

	vec2_t Normalized() const
	{
		return *this * (1.0 / sqrt(X * X + Y * Y + Z * Z));
	}

	vec2_t& Normalize()
	{
		return *this = Normalized();
	}

	Type Dot(const vec2_t& Other) const
	{
		return (X * Other.X + Y * Other.Y);
	}

	Type Length(const vec2_t& Other) const
	{
		return sqrt(pow(Other.X - X, 2) + pow(Other.Y - Y, 2));
	}
};

template <typename Type>
struct vec3_t
{
	Type X;
	Type Y;
	Type Z;

	vec3_t() {}
	vec3_t(Type A) : X(A), Y(A), Z(A) {}
	vec3_t(Type InX, Type InY, Type InZ) : X(InX), Y(InY), Z(InZ) {}
	vec3_t(const vec2_t<Type>& A, Type B) : X(A.X), Y(A.Y), Z(B) {}
	vec3_t(Type A, const vec2_t<Type>& B) : X(A), Y(B.X), Z(B.Y) {}
	vec3_t(const vec3_t& Base) : X(Base.X), Y(Base.Y), Z(Base.Z) {}
	vec3_t(vec3_t&& Base) :
		X(std::move(Base.X)),
		Y(std::move(Base.Y)),
		Z(std::move(Base.Z)) {}

	vec2_t<Type> XX() { return vec2_t<Type>(X, X); }
	vec2_t<Type> XY() { return vec2_t<Type>(X, Y); }
	vec2_t<Type> XZ() { return vec2_t<Type>(X, Z); }
	vec2_t<Type> YX() { return vec2_t<Type>(Y, X); }
	vec2_t<Type> YY() { return vec2_t<Type>(Y, Y); }
	vec2_t<Type> YZ() { return vec2_t<Type>(Y, Z); }
	vec2_t<Type> ZX() { return vec2_t<Type>(Z, X); }
	vec2_t<Type> ZY() { return vec2_t<Type>(Z, Y); }
	vec2_t<Type> ZZ() { return vec2_t<Type>(Z, Z); }

	vec3_t& operator=(const vec3_t& Other)
	{
		X = Other.X;
		Y = Other.Y;
		Z = Other.Z;
		return *this;
	}

	vec3_t& operator=(vec3_t&& Other)
	{
		X = std::move(Other.X);
		Y = std::move(Other.Y);
		Z = std::move(Other.Z);
		return *this;
	}

	vec3_t operator+(const vec3_t& Other) const
	{
		return vec3_t(X + Other.X, Y + Other.Y, Z + Other.Z);
	}

	vec3_t operator-() const
	{
		return vec3_t(-X, -Y, -Z);
	}

	vec3_t operator-(const vec3_t& Other) const
	{
		return vec3_t(X - Other.X, Y - Other.Y, Z - Other.Z);
	}

	vec3_t operator*(const Type& Scalar) const
	{
		return vec3_t(X * Scalar, Y * Scalar, Z * Scalar);
	}

	friend vec3_t operator*(const Type& Scalar, const vec3_t& Other)
	{
		return Other * Scalar;
	}

	vec3_t operator*(const vec3_t& Other) const
	{
		return vec3_t(X * Other.X, Y * Other.Y, Z * Other.Z);
	}

	vec3_t operator/(const Type& Scalar) const
	{
		const Type Factor = 1.0 / Scalar;
		return vec3_t(X * Factor, Y * Factor, Z * Factor);
	}

	vec3_t operator/(const vec3_t& Other) const
	{
		return vec3_t(X / Other.X, Y / Other.Y, Z / Other.Z);
	}

	vec3_t& operator+=(const vec3_t& Other)
	{
		return *this = *this + Other;
	}

	vec3_t& operator-=(const vec3_t& Other)
	{
		return *this = *this - Other;
	}

	vec3_t& operator*=(const Type& Scalar)
	{
		return *this = *this * Scalar;
	}

	vec3_t& operator*=(const vec3_t& Other)
	{
		return *this = *this * Other;
	}

	vec3_t& operator/=(const Type& Scalar)
	{
		return *this = *this / Scalar;
	}

	vec3_t& operator/=(const vec3_t& Other)
	{
		return *this = *this / Other;
	}

	vec3_t Clamped(const Type& Min, const Type& Max) const
	{
		return vec3_t(Math::Clamp(Min, Max, X), Math::Clamp(Min, Max, Y), Math::Clamp(Min, Max, Z));
	}

	vec3_t Clamped(const vec3_t& Min, const vec3_t& Max) const
	{
		return vec3_t(Math::Clamp(Min.X, Max.X, X), Math::Clamp(Min.Y, Max.Y, Y), Math::Clamp(Min.Z, Max.Z, Z));
	}

	vec3_t& Clamp(const Type& Min, const Type& Max)
	{
		return *this = Clamped(Min, Max);
	}

	vec3_t& Clamp(const vec3_t& Min, const vec3_t& Max)
	{
		return *this = Clamped(Min, Max);
	}

	vec3_t Normalized() const
	{
		return *this * (1.0 / sqrt(X * X + Y * Y + Z * Z));
	}

	vec3_t& Normalize()
	{
		return *this = Normalized();
	}

	Type Dot(const vec3_t& Other) const
	{
		return (X * Other.X + Y * Other.Y + Z * Other.Z);
	}

	vec3_t Cross(const vec3_t& Other) const
	{
		return vec3_t(Y * Other.Z - Z * Other.Y, Z * Other.X - X * Other.Z, X * Other.Y - Y * Other.X);
	}

	Type Length(const vec3_t& Other) const
	{
		return sqrt(pow(Other.X - X, 2) + pow(Other.Y - Y, 2) + pow(Other.Z - Z, 2));
	}

	vec3_t Reflect(const vec3_t& Normal) const
	{
		return *this - this->Dot(Normal) * 2 * Normal;
	}
};

template <typename Type>
struct vec4_t
{
	Type X;
	Type Y;
	Type Z;
	Type W;

	vec4_t() {}
	vec4_t(Type A) : X(A), Y(A), Z(A), W(A) {}
	vec4_t(Type InX, Type InY, Type InZ, Type InW) : X(InX), Y(InY), Z(InZ), W(InW) {}
	vec4_t(const vec2_t<Type>& A, Type B, Type C) : X(A.X), Y(A.Y), Z(B), W(C) {}
	vec4_t(Type A, Type B, const vec2_t<Type>& C) : X(A), Y(B), Z(C.X), W(C.Y) {}
	vec4_t(const vec2_t<Type>& A, const vec2_t<Type>& B) : X(A.X), Y(A.Y), Z(B.X), W(B.Y) {}
	vec4_t(const vec3_t<Type>& A, Type B) : X(A.X), Y(A.Y), Z(A.Z), W(B) {}
	vec4_t(Type A, const vec3_t<Type>& B) : X(A), Y(B.X), Z(B.Y), W(B.Z) {}
	vec4_t(const vec4_t& Base) : X(Base.X), Y(Base.Y), Z(Base.Y), W(Base.W) {}
	vec4_t(vec4_t&& Base) :
		X(std::move(Base.X)),
		Y(std::move(Base.Y)),
		Z(std::move(Base.Z)),
		W(std::move(Base.W)) {}

	vec2_t<Type> XX() { return vec2_t<Type>(X, X); }
	vec2_t<Type> XY() { return vec2_t<Type>(X, Y); }
	vec2_t<Type> XZ() { return vec2_t<Type>(X, Z); }
	vec2_t<Type> XW() { return vec2_t<Type>(X, W); }
	vec2_t<Type> YX() { return vec2_t<Type>(Y, X); }
	vec2_t<Type> YY() { return vec2_t<Type>(Y, Y); }
	vec2_t<Type> YZ() { return vec2_t<Type>(Y, Z); }
	vec2_t<Type> YW() { return vec2_t<Type>(Y, W); }
	vec2_t<Type> ZX() { return vec2_t<Type>(Z, X); }
	vec2_t<Type> ZY() { return vec2_t<Type>(Z, Y); }
	vec2_t<Type> ZZ() { return vec2_t<Type>(Z, Z); }
	vec2_t<Type> ZW() { return vec2_t<Type>(Z, W); }

	vec3_t<Type> XXX() { return vec3_t<Type>(X, X, X); }
	vec3_t<Type> XXY() { return vec3_t<Type>(X, X, Y); }
	vec3_t<Type> XXZ() { return vec3_t<Type>(X, X, Z); }
	vec3_t<Type> XXW() { return vec3_t<Type>(X, X, W); }
	vec3_t<Type> XYX() { return vec3_t<Type>(X, Y, X); }
	vec3_t<Type> XYY() { return vec3_t<Type>(X, Y, Y); }
	vec3_t<Type> XYZ() { return vec3_t<Type>(X, Y, Z); }
	vec3_t<Type> XYW() { return vec3_t<Type>(X, Y, W); }
	vec3_t<Type> XZX() { return vec3_t<Type>(X, Z, X); }
	vec3_t<Type> XZY() { return vec3_t<Type>(X, Z, Y); }
	vec3_t<Type> XZZ() { return vec3_t<Type>(X, Z, Z); }
	vec3_t<Type> XZW() { return vec3_t<Type>(X, Z, W); }
	vec3_t<Type> XWX() { return vec3_t<Type>(X, W, X); }
	vec3_t<Type> XWY() { return vec3_t<Type>(X, W, Y); }
	vec3_t<Type> XWZ() { return vec3_t<Type>(X, W, Z); }
	vec3_t<Type> XWW() { return vec3_t<Type>(X, W, W); }

	vec3_t<Type> YXX() { return vec3_t<Type>(Y, X, X); }
	vec3_t<Type> YXY() { return vec3_t<Type>(Y, X, Y); }
	vec3_t<Type> YXZ() { return vec3_t<Type>(Y, X, Z); }
	vec3_t<Type> YXW() { return vec3_t<Type>(Y, X, W); }
	vec3_t<Type> YYX() { return vec3_t<Type>(Y, Y, X); }
	vec3_t<Type> YYY() { return vec3_t<Type>(Y, Y, Y); }
	vec3_t<Type> YYZ() { return vec3_t<Type>(Y, Y, Z); }
	vec3_t<Type> YYW() { return vec3_t<Type>(Y, Y, W); }
	vec3_t<Type> YZX() { return vec3_t<Type>(Y, Z, X); }
	vec3_t<Type> YZY() { return vec3_t<Type>(Y, Z, Y); }
	vec3_t<Type> YZZ() { return vec3_t<Type>(Y, Z, Z); }
	vec3_t<Type> YZW() { return vec3_t<Type>(Y, Z, W); }
	vec3_t<Type> YWX() { return vec3_t<Type>(Y, W, X); }
	vec3_t<Type> YWY() { return vec3_t<Type>(Y, W, Y); }
	vec3_t<Type> YWZ() { return vec3_t<Type>(Y, W, Z); }
	vec3_t<Type> YWW() { return vec3_t<Type>(Y, W, W); }

	vec3_t<Type> ZXX() { return vec3_t<Type>(Z, X, X); }
	vec3_t<Type> ZXY() { return vec3_t<Type>(Z, X, Y); }
	vec3_t<Type> ZXZ() { return vec3_t<Type>(Z, X, Z); }
	vec3_t<Type> ZXW() { return vec3_t<Type>(Z, X, W); }
	vec3_t<Type> ZYX() { return vec3_t<Type>(Z, Y, X); }
	vec3_t<Type> ZYY() { return vec3_t<Type>(Z, Y, Y); }
	vec3_t<Type> ZYZ() { return vec3_t<Type>(Z, Y, Z); }
	vec3_t<Type> ZYW() { return vec3_t<Type>(Z, Y, W); }
	vec3_t<Type> ZZX() { return vec3_t<Type>(Z, Z, X); }
	vec3_t<Type> ZZY() { return vec3_t<Type>(Z, Z, Y); }
	vec3_t<Type> ZZZ() { return vec3_t<Type>(Z, Z, Z); }
	vec3_t<Type> ZZW() { return vec3_t<Type>(Z, Z, W); }
	vec3_t<Type> ZWX() { return vec3_t<Type>(Z, W, X); }
	vec3_t<Type> ZWY() { return vec3_t<Type>(Z, W, Y); }
	vec3_t<Type> ZWZ() { return vec3_t<Type>(Z, W, Z); }
	vec3_t<Type> ZWW() { return vec3_t<Type>(Z, W, W); }

	vec3_t<Type> WXX() { return vec3_t<Type>(W, X, X); }
	vec3_t<Type> WXY() { return vec3_t<Type>(W, X, Y); }
	vec3_t<Type> WXZ() { return vec3_t<Type>(W, X, Z); }
	vec3_t<Type> WXW() { return vec3_t<Type>(W, X, W); }
	vec3_t<Type> WYX() { return vec3_t<Type>(W, Y, X); }
	vec3_t<Type> WYY() { return vec3_t<Type>(W, Y, Y); }
	vec3_t<Type> WYZ() { return vec3_t<Type>(W, Y, Z); }
	vec3_t<Type> WYW() { return vec3_t<Type>(W, Y, W); }
	vec3_t<Type> WZX() { return vec3_t<Type>(W, Z, X); }
	vec3_t<Type> WZY() { return vec3_t<Type>(W, Z, Y); }
	vec3_t<Type> WZZ() { return vec3_t<Type>(W, Z, Z); }
	vec3_t<Type> WZW() { return vec3_t<Type>(W, Z, W); }
	vec3_t<Type> WWX() { return vec3_t<Type>(W, W, X); }
	vec3_t<Type> WWY() { return vec3_t<Type>(W, W, Y); }
	vec3_t<Type> WWZ() { return vec3_t<Type>(W, W, Z); }
	vec3_t<Type> WWW() { return vec3_t<Type>(W, W, W); }

	vec4_t& operator=(const vec4_t& Other)
	{
		X = Other.X;
		Y = Other.Y;
		Z = Other.Z;
		W = Other.W;
		return *this;
	}

	vec4_t& operator=(vec4_t&& Other)
	{
		X = std::move(Other.X);
		Y = std::move(Other.Y);
		Z = std::move(Other.Z);
		W = std::move(Other.W);
		return *this;
	}

	vec4_t operator+(const vec4_t& Other) const
	{
		return vec4_t(X + Other.X, Y + Other.Y, Z + Other.Z, W + Other.W);
	}

	vec4_t operator-() const
	{
		return vec4_t(-X, -Y, -Z, -W);
	}

	vec4_t operator-(const vec4_t& Other) const
	{
		return vec4_t(X - Other.X, Y - Other.Y, Z - Other.Z, W - Other.W);
	}

	vec4_t operator*(const Type& Scalar) const
	{
		return vec4_t(X * Scalar, Y * Scalar, Z * Scalar, W * Scalar);
	}

	friend vec4_t operator*(const Type& Scalar, const vec4_t& Other)
	{
		return Other * Scalar;
	}

	vec4_t operator*(const vec4_t& Other) const
	{
		return vec4_t(X * Other.X, Y * Other.Y, Z * Other.Z, W * Other.W);
	}

	vec4_t operator/(const Type& Scalar) const
	{
		const Type Factor = 1.0 / Scalar;
		return vec4_t(X * Factor, Y * Factor, Z * Factor, W * Factor);
	}

	vec4_t operator/(const vec4_t& Other) const
	{
		return vec4_t(X / Other.X, Y / Other.Y, Z / Other.Z, W / Other.W);
	}

	vec4_t& operator+=(const vec4_t& Other)
	{
		return *this = *this + Other;
	}

	vec4_t& operator-=(const vec4_t& Other)
	{
		return *this = *this - Other;
	}

	vec4_t& operator*=(const Type& Scalar)
	{
		return *this = *this * Scalar;
	}

	vec4_t& operator*=(const vec4_t& Other)
	{
		return *this = *this * Other;
	}

	vec4_t& operator/=(const Type& Scalar)
	{
		return *this = *this / Scalar;
	}

	vec4_t& operator/=(const vec4_t& Other)
	{
		return *this = *this / Other;
	}

	vec4_t Clamped(const Type& Min, const Type& Max) const
	{
		return vec4_t(Math::Clamp(Min, Max, X), Math::Clamp(Min, Max, Y), Math::Clamp(Min, Max, Z), Math::Clamp(Min, Max, W));
	}

	vec4_t Clamped(const vec4_t& Min, const vec4_t& Max) const
	{
		return vec4_t(Math::Clamp(Min.X, Max.X, X), Math::Clamp(Min.Y, Max.Y, Y), Math::Clamp(Min.Z, Max.Z, Z), Math::Clamp(Min.W, Max.W, W));
	}

	vec4_t& Clamp(const Type& Min, const Type& Max)
	{
		return *this = Clamped(Min, Max);
	}

	vec4_t& Clamp(const vec4_t& Min, const vec4_t& Max)
	{
		return *this = Clamped(Min, Max);
	}

	vec4_t Normalized() const
	{
		return *this * (1 / sqrt(X * X + Y * Y + Z * Z + W * W));
	}

	vec4_t& Normalize()
	{
		return *this = Normalized();
	}

	Type Dot(const vec4_t& Other) const
	{
		return (X * Other.X + Y * Other.Y + Z * Other.Z + W * Other.W);
	}

	Type Length(const vec4_t& Other) const
	{
		return sqrt(pow(Other.X - X, 2) + pow(Other.Y - Y, 2) + pow(Other.Z - Z, 2) + pow(Other.W - W ,2));
	}
};

template <typename Type>
struct mat4_t
{
	Type Matrix[4][4];

	mat4_t() {}
	mat4_t(const Type& Diagonal) : mat4_t(
		vec4(Diagonal, 0, 0, 0),
		vec4(0, Diagonal, 0, 0),
		vec4(0, 0, Diagonal, 0),
		vec4(0, 0, 0, Diagonal)) {}
	mat4_t(const mat4_t& Base) { std::copy(&Base.Matrix[0][0], &Base.Matrix[0][0] + (4 * 4), Matrix); }
	mat4_t(mat4_t&& Base) { Matrix = std::move(Base.Matrix); }
	mat4_t(const vec4_t<Type>& A, const vec4_t<Type>& B, const vec4_t<Type>& C, const vec4_t<Type>& D)
	{
		SetRow(0, A);
		SetRow(1, B);
		SetRow(2, C);
		SetRow(3, D);
	}

	mat4_t& Clear()
	{
		return *this = mat4_t(1);
	}

	mat4_t& SetIdentity()
	{
		return *this = mat4_t(1);
	}

	void SetRow(size_t Row, const vec4_t<Type>& Value)
	{
		Matrix[Row][0] = Value.X;
		Matrix[Row][1] = Value.Y;
		Matrix[Row][2] = Value.Z;
		Matrix[Row][3] = Value.W;
	}

	vec4_t<Type> operator*(const vec4_t<Type>& Vector) const
	{
		vec4_t<Type> Result;

		for (size_t Row = 0; Row < 4; Row++)
		{
			Result.X += Matrix[Row][0] * Vector.X;
			Result.Y += Matrix[Row][1] * Vector.Y;
			Result.Z += Matrix[Row][2] * Vector.Z;
			Result.W += Matrix[Row][3] * Vector.W;
		}

		return Result;
	}

	mat4_t& LookAt(const vec3_t<Type>& Position, const vec3_t<Type>& Center, const vec3_t<Type>& Up)
	{
		vec3_t<Type> const f((Center - Position).Normalized());
		vec3_t<Type> const s(f.Cross(Up).Normalized());
		vec3_t<Type> const u(s.Cross(f));

		SetIdentity();

		Matrix[0][0] = s.X;
		Matrix[1][0] = s.Y;
		Matrix[2][0] = s.Z;
		Matrix[0][1] = u.X;
		Matrix[1][1] = u.Y;
		Matrix[2][1] = u.Z;
		Matrix[0][2] = -f.X;
		Matrix[1][2] = -f.Y;
		Matrix[2][2] = -f.Z;
		Matrix[3][0] = -s.Dot(Position);
		Matrix[3][1] = -u.Dot(Position);
		Matrix[3][2] = f.Dot(Position);

		return *this;
	}
};

struct Ray
{
	vec3 Begin;
	vec3 Direction;

	Ray(vec3 B, vec3 D) : Begin(B), Direction(D) { Direction.Normalize(); }
};

struct PointLight
{
	vec3 Position;
	vec3 Color;

	PointLight(vec3 P, vec3 C) : Position(P), Color(C) {}
};

struct Sphere
{
	vec3 Position;
	vec4 Color;
	double Radius;

	Sphere(vec3 P, vec4 C, double R) : Position(P), Color(C), Radius(R) {}

	bool Intersect(const Ray& R, vec3& Normal, vec3& Hitpoint) const
	{	
		vec3 V = Position - R.Begin;
		double Dot = V.Dot(R.Direction);
		double Distance = (V - R.Direction * Dot).Length({0});

		Hitpoint = R.Direction * (Dot - sqrt(Radius * Radius - Distance * Distance)) + R.Begin;
		Normal = (Hitpoint - Position).Normalize();

		return Distance <= Radius && Distance >= 0.0;
	}

	vec3 LightFunc(PointLight Light, vec3 Normal, vec3 Hitpoint, vec3 Ambient, vec3 Camera)
	{
		vec3 LightDir = (Hitpoint - Light.Position).Normalized();
		vec3 ViewDir = (Camera - Hitpoint).Normalized();
		vec3 ReflectDir = LightDir.Reflect(Normal).Normalized();
		float DiffuseFactor = Math::Max(0.0f, Normal.Dot(-LightDir));
		float SpecularFactor = pow(Math::Max(0.0f, ViewDir.Dot(ReflectDir)), 32);

		vec3 AmbientColor = Color.XYZ() * Ambient;
		vec3 DiffuseColor = Light.Color * Color.XYZ() * DiffuseFactor;
		vec3 SpecularColor = Light.Color * Color.XYZ() * SpecularFactor;

		return (AmbientColor + DiffuseColor + SpecularColor).Clamped(0.0, 1.0);
	}
};










