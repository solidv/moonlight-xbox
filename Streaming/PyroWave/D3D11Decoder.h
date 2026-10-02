#pragma once

// D3D11 port of PyroWave::Decoder (third_party/pyrowave, MIT).
// CPU bitstream handling is ported near-verbatim from pyrowave_decoder.cpp /
// pyrowave_common.cpp; the GPU layer maps Granite/Vulkan onto D3D11 compute.
// See docs/pyrowave-decoder-notes.md for the full mapping, including why the
// wavelet storage is 5 per-level textures here instead of one mipped image.

#include <cstdint>
#include <d3d11.h>
#include <vector>
#include <wrl/client.h>

namespace moonlight_xbox_dx {
namespace PyroWaveD3D11 {

// Bitstream structures, identical layout to pyrowave_common.hpp
struct BitstreamHeader {
	uint16_t ballot;
	uint16_t payload_words : 12;
	uint16_t sequence : 3;
	uint16_t extended : 1;
	uint32_t quant_code : 8;
	uint32_t block_index : 24;
};
static_assert(sizeof(BitstreamHeader) == 8, "BitstreamHeader is not 8 bytes.");

struct BitstreamSequenceHeader {
	uint32_t width_minus_1 : 14;
	uint32_t height_minus_1 : 14;
	uint32_t sequence : 3;
	uint32_t extended : 1;
	uint32_t total_blocks : 24;
	uint32_t code : 2;
	uint32_t chroma_resolution : 1;
	uint32_t color_primaries : 1;
	uint32_t transfer_function : 1;
	uint32_t ycbcr_transform : 1;
	uint32_t ycbcr_range : 1;
	uint32_t chroma_siting : 1;
};
static_assert(sizeof(BitstreamSequenceHeader) == 8, "BitstreamSequenceHeader is not 8 bytes.");

// Colorimetry parsed from the most recent sequence header (enum values match
// pyrowave_common.hpp: BT709=0/BT2020=1, SDR=0/PQ=1, full=0/limited=1).
struct SequenceColorimetry {
	int colorPrimaries = 0;
	int transferFunction = 0;
	int ycbcrTransform = 0;
	int ycbcrRange = 0;
	int chromaSiting = 0;
	bool valid = false;
};

// One contiguous byte range of a decode unit's buffer chain, tagged with the
// byte offset it occupies in the GPU payload buffer. That buffer is laid out
// exactly like the decode unit (transport framing included), so the parser
// records word offsets into this layout and no CPU staging copy is needed.
struct PayloadSegment {
	const uint8_t *data;
	uint32_t size;
	uint32_t offset;
};

class Decoder {
  public:
	// chroma444: true for 4:4:4 streams (all our targets), false for 4:2:0.
	// Fails (returns false) on shader load or resource creation errors.
	bool Init(ID3D11Device *device, int width, int height, bool chroma444);

	// Parse `size` bytes of one transport chunk starting at
	// (startSegment, startOffset) in the segment list. Packets may span
	// segments. Only 8-byte headers are ever copied: payload bytes stay in the
	// depacketizer's buffers until Decode() uploads them to the GPU.
	// allowTruncated: the chunk is known to be cut short (partial decode unit),
	// so running out of data mid-packet is expected rather than an error. Whole
	// packets ahead of the cut are still decoded.
	bool PushPacket(const PayloadSegment *segments, size_t segmentCount,
	                size_t startSegment, size_t startOffset, size_t size,
	                bool allowTruncated = false);

	// Partial-frame readiness, ported from upstream's decode_is_ready()
	// (pyrowave_decoder.cpp 89f7e47+). `pristineBands` counts wavelet bands
	// from the coarsest level, and `minimumPacketRatio` is the fraction of the
	// transmitted blocks that must have arrived (0 disables the floor).
	// `activeBlockMask` is the optional encoder sideband (bit b of word b/32
	// set = block b was expected; masked-out blocks are ignored), or null to
	// require every block that the bands cover.
	bool DecodeIsReady(bool allowPartialFrame) const;
	bool DecodeIsReady(bool allowPartialFrame, int pristineBands, float minimumPacketRatio,
	                   const uint32_t *activeBlockMask, size_t maskWordCount) const;

	// Policy applied by DecodeIsReady(allowPartialFrame) and by callers that
	// supply the active-block mask: bands counted from the coarsest level, where
	// 3 covers decomposition levels 4 and 3 — exactly the head Sunshine protects
	// with 50% FEC (FEC_PROTECTED_BANDS). The ratio floor is off: prefixes far
	// below upstream's 90% default decode to a useful blur.
	static constexpr int kPartialPristineBands = 3;
	static constexpr float kPartialMinimumPacketRatio = 0.0f;

	// Maps the payload buffer and writes the decode unit (the segments, in
	// decode-unit order) directly into it, then records dequant + iDWT on the
	// given context. There is no CPU staging copy of the payload.
	// planeUavs are the three output plane UAVs (Y, Cb, Cr), full frame
	// resolution each for 4:4:4. They MUST be created with ViewDimension
	// TEXTURE2DARRAY (ArraySize 1): the idwt kernel is built with
	// OUTPUT_LAYERED and declares RWTexture2DArray. Caller must hold the
	// device-context lock.
	bool Decode(ID3D11DeviceContext *ctx, ID3D11UnorderedAccessView *const planeUavs[3],
	            const PayloadSegment *segments, size_t segmentCount, size_t totalBytes);

	// Reset all sequence state (stream restart).
	void Clear();

	// Non-blocking GPU decode timing. Decode() brackets its uploads and
	// dispatches with timestamp queries into a small ring; this polls the
	// oldest outstanding measurement WITHOUT flushing or stalling. Returns
	// true and the GPU time of some recent frame (1-3 frames old) when one
	// is ready. Call on the immediate context under the same lock as
	// Decode(). Disjoint results (GPU clock changed) are consumed silently.
	bool PollGpuTimeMs(ID3D11DeviceContext *ctx, double *outMs);

	const SequenceColorimetry &Colorimetry() const {
		return m_colorimetry;
	}

	// Blocks decoded so far vs the transmitted count from the sequence header.
	// Their ratio is how much of a partial frame actually arrived (used to name
	// the automatic partial-frame captures).
	int DecodedBlocks() const { return m_decodedBlocks; }
	int TotalBlocksInSequence() const { return m_totalBlocksInSequence; }

	int AlignedWidth() const { return m_alignedWidth; }
	int AlignedHeight() const { return m_alignedHeight; }

  private:
	static constexpr int kLevels = 5;          // DecompositionLevels
	static constexpr int kComponents = 3;      // NumComponents
	static constexpr int kBandsPerLevel = 4;   // NumFrequencyBandsPerLevel
	static constexpr int kAlignment = 1 << kLevels;
	static constexpr int kMinimumImageSize = 4 << kLevels;
	static constexpr uint32_t kSequenceCountMask = 0x7;

	struct BlockInfo {
		int blockOffset8x8;
		int blockStride8x8;
		int blockOffset32x32;
		int blockStride32x32;
		// BlocksX32 * blocksY32 for this band (upstream's block_count_32x32).
		int blockCount32x32;
	};

	// Records the new block's word offset for Decode()'s upload; payload bytes
	// are not copied here (duplicates and invalid packets append nothing).
	bool DecodePacket(const BitstreamHeader &header, uint32_t payloadWordOffset);
	// Port of upstream's Impl::has_pristine_bands(): true when every block that
	// the given bands cover has data. Null blocks are rare in the coarsest
	// bands, but when one is absent this cannot tell it from a lost block —
	// that is what the activeBlockMask sideband (and the DecodeIsReady
	// fallback) are for.
	bool HasPristineBands(int bands, const uint32_t *activeBlockMask, size_t maskWordCount) const;
	void InitBlockMeta();
	bool CreateResources(ID3D11Device *device);
	bool EnsurePayloadBuffer(size_t requiredBytes);
	void Dequant(ID3D11DeviceContext *ctx);
	void Idwt(ID3D11DeviceContext *ctx, ID3D11UnorderedAccessView *const planeUavs[3]);

	int LevelWidth(int level) const {
		return (m_alignedWidth / 2) >> level;
	}
	int LevelHeight(int level) const {
		return (m_alignedHeight / 2) >> level;
	}

	Microsoft::WRL::ComPtr<ID3D11Device> m_device;

	// Shaders
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_dequantShader;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_idwtShader;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_idwtDcShader;

	// Wavelet band storage: one texture per decomposition level, 12 slices
	// (4 bands x 3 components, slice = 4*comp + band), R16F, single mip.
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_bandTex[kLevels];
	// 4-slice views per component per level
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_bandSrv[kComponents][kLevels];
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_bandUav[kComponents][kLevels];
	// single-slice LL views (iDWT output for the next-finer level)
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_llUav[kComponents][kLevels];

	// Immutable per-dispatch constant buffers (frame-invariant contents)
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_dequantCb[kComponents][kLevels][kBandsPerLevel];
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_idwtCb[kLevels];

	Microsoft::WRL::ComPtr<ID3D11SamplerState> m_mirrorSampler;

	// GPU decode timing (see PollGpuTimeMs)
	static constexpr int kGpuTimerSlots = 8;
	struct GpuTimerSlot {
		Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
		Microsoft::WRL::ComPtr<ID3D11Query> tsBegin;
		Microsoft::WRL::ComPtr<ID3D11Query> tsEnd;
		bool pending = false;
	};
	GpuTimerSlot m_gpuTimers[kGpuTimerSlots];
	int m_gpuTimerWrite = 0;
	int m_gpuTimerRead = 0;

	// Bitstream GPU buffers
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_offsetsBuffer;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_offsetsSrv;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_payloadBuffer;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_payloadU8Srv, m_payloadU16Srv, m_payloadU32Srv;
	size_t m_payloadBufferBytes = 0;

	// CPU-side state (mirrors upstream Impl)
	std::vector<uint32_t> m_offsetsCpu;
	int m_decodedBlocks = 0;
	// Whether any packet of the current sequence started at a block index at
	// or past m_coarseBlockEnd. Kept as a DecodeIsReady fallback: it also
	// accepts frames whose coarsest blocks include null blocks that were never
	// transmitted, which the explicit per-band check cannot see.
	bool m_sawBlockBeyondCoarse = false;
	int m_totalBlocksInSequence = 0;
	uint32_t m_lastSeq = UINT32_MAX;
	bool m_decodedFrameForCurrentSequence = false;

	BlockInfo m_blockMeta[kComponents][kLevels][kBandsPerLevel] = {};
	int m_blockCount8x8 = 0;
	int m_blockCount32x32 = 0;
	// First block index finer than level 3; blocks [0, this) are the coarse
	// levels whose complete arrival gates partial-frame decode.
	int m_coarseBlockEnd = 0;

	int m_width = 0, m_height = 0;
	int m_alignedWidth = 0, m_alignedHeight = 0;
	bool m_chroma444 = true;
	SequenceColorimetry m_colorimetry;
};

} // namespace PyroWaveD3D11
} // namespace moonlight_xbox_dx
