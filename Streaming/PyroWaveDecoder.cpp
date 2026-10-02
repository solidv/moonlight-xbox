#include "pch.h"
#include "PyroWaveDecoder.h"
#include "FFmpegDecoder.h" // context lock + MLFrameData
#include "Pacer.h"
#include "../Common/DeviceResources.h"
#include <Utils.hpp>

extern "C" {
#include <Limelight.h>
#include <libavutil/frame.h>
#include <libavutil/time.h>
}

namespace moonlight_xbox_dx {

namespace {

// Sanity cap for the active-block mask header: 4096 words covers 128K blocks,
// far beyond any real frame (a 4K 4:4:4 frame's protected bands are a few tens
// of words).
constexpr uint32_t kMaxActiveMaskWords = 4096;

// Byte cursor over the decode unit's segment list, used only for the transport
// framing's 4-byte values. Packet bytes are parsed in place by the decoder.
struct SegmentCursor {
	const PyroWaveD3D11::PayloadSegment *segments;
	size_t count;
	size_t index = 0;
	size_t offset = 0;
	size_t remaining = 0;

	// Advances past empty segments; false when the segment list is exhausted.
	bool Normalize() {
		while (index < count && offset == segments[index].size) {
			offset = 0;
			++index;
		}
		return index < count;
	}

	bool ReadU32(uint32_t &value) {
		if (remaining < 4)
			return false;
		uint8_t bytes[4];
		for (size_t i = 0; i < 4; i++) {
			if (!Normalize())
				return false;
			bytes[i] = segments[index].data[offset++];
		}
		remaining -= 4;
		memcpy(&value, bytes, sizeof(value));
		return true;
	}

	bool Skip(size_t n) {
		if (n > remaining)
			return false;
		remaining -= n;
		while (n > 0) {
			if (!Normalize())
				return false;
			size_t available = segments[index].size - offset;
			size_t take = available < n ? available : n;
			offset += take;
			n -= take;
		}
		return true;
	}
};

} // namespace

PyroWaveDecoder &PyroWaveDecoder::instance() {
	static PyroWaveDecoder inst;
	return inst;
}

void PyroWaveDecoder::CompleteInitialization(const std::shared_ptr<DX::DeviceResources> &res, STREAM_CONFIGURATION *config) {
	m_deviceResources = res;
	// The negotiated colorSpace/colorRange are requests PyroWave's host-side
	// scaler ignores; ApplyFrameColor sets the AVFrame color fields from what
	// it actually sends.
	(void)config;
}

int PyroWaveDecoder::Init(int videoFormat_, int width_, int height_, int redrawRate) {
	(void)redrawRate; // Pacer is configured in FFMpegDecoder::CompleteInitialization

	videoFormat = videoFormat_;
	width = width_;
	height = height_;
	m_LastFrameNumber = 0;
	m_StreamEpochQpc = 0;
	// The host advertised the sideband in the DESCRIBE reply and common-c
	// requested it back in the ANNOUNCE, so the host feature bit is also what
	// says whether this session's frames carry the mask.
	m_activeBlockSideband = (LiGetHostFeatureFlags() & LI_FF_PYROWAVE_ACTIVE_BLOCK_MASK) != 0;
	m_activeMask.clear();
	// Don't let a capture armed at the end of one session fire in the next.
	m_captureFramesRemaining.store(0, std::memory_order_relaxed);
	// Auto-save the first few partial frames each stream for offline review.
	m_partialCapturesRemaining.store(kPartialCapturesPerStream, std::memory_order_relaxed);

	m_retiredPools.clear();

	bool chroma444 = (videoFormat & (VIDEO_FORMAT_PYROWAVE_444 | VIDEO_FORMAT_PYROWAVE10_444)) != 0;

	auto *device = m_deviceResources->GetD3DDevice();
	m_decoder = std::make_unique<PyroWaveD3D11::Decoder>();
	if (!m_decoder->Init(device, width, height, chroma444)) {
		Utils::Log("PyroWave live: decoder Init failed\n");
		m_decoder.reset();
		return -1;
	}
	// 6 sets: DUs can arrive in a burst right after loss recovery while the
	// renderer is still catching up (observed pool exhaustion with 4 at
	// 4K120 during torture testing).
	m_pool = std::make_unique<PyroWaveD3D11::FramePool>();
	if (!m_pool->Init(device, width, height, chroma444, 6)) {
		Utils::Log("PyroWave live: FramePool Init failed\n");
		m_decoder.reset();
		m_pool.reset();
		return -1;
	}

	m_active = true;
	Utils::Logf("PyroWave live: init %dx%d %s, format 0x%04x%s\n",
	            width, height, chroma444 ? "4:4:4" : "4:2:0", videoFormat,
	            m_activeBlockSideband ? ", active-block sideband" : "");
	return 0;
}

void PyroWaveDecoder::Cleanup() {
	m_active = false;
	m_decoder.reset();
	if (m_pool) {
		// Frames wrapped around this pool's sets may still sit in Pacer's
		// queue until it drains; keep the pool alive until the next Init.
		m_retiredPools.push_back(std::move(m_pool));
	}
	m_LastFrameNumber = 0;

	Pacer::instance().deinit();

	Utils::Log("PyroWaveDecoder::Cleanup\n");
}

void PyroWaveDecoder::ApplyFrameColor(AVFrame *frame) {
	// PyroWave's host-side scaler always emits full-range YCbCr: BT.709 for
	// SDR, BT.2020 NCL PQ for HDR. The bitstream can't signal that, and the
	// negotiated colorSpace/colorRange are requests the host doesn't honor,
	// so decode what the host actually sends.
	frame->color_range = AVCOL_RANGE_JPEG;

	bool hdr = (videoFormat & VIDEO_FORMAT_MASK_10BIT) && LiGetCurrentHostDisplayHdrMode();
	if (hdr) {
		frame->colorspace = AVCOL_SPC_BT2020_NCL;
		frame->color_primaries = AVCOL_PRI_BT2020;
		frame->color_trc = AVCOL_TRC_SMPTE2084;
	} else {
		frame->colorspace = AVCOL_SPC_BT709;
		frame->color_primaries = AVCOL_PRI_BT709;
		frame->color_trc = AVCOL_TRC_BT709;
	}
}

int PyroWaveDecoder::CaptureFrames(int count) {
	if (!m_active || count <= 0)
		return 0;
	m_captureFramesRemaining.store(count, std::memory_order_relaxed);
	return count;
}

// Copies the DU and writes it on a background task; storage I/O on the decode
// thread would stall it long enough to drop frames, and the resulting loss
// would change what the capture is trying to study.
void PyroWaveDecoder::WriteCaptureAsync(int frameNumber, int lostPercent) {
	// The file must hold exactly what the host framed, so flatten the segment
	// view back into one buffer. Capture is a dev tool, so the copy is fine;
	// only the write itself needs to stay off the decode thread.
	auto copy = std::make_shared<std::vector<uint8_t>>();
	size_t length = 0;
	for (const auto &segment : m_segments)
		length += segment.size;
	copy->reserve(length);
	for (const auto &segment : m_segments)
		copy->insert(copy->end(), segment.data, segment.data + segment.size);
	Concurrency::create_task([this, copy, frameNumber, lostPercent] {
		// An exception escaping a discarded task takes the process down at
		// task destruction, and this one touches WinRT storage APIs.
		try {
			WriteCapture(copy->data(), copy->size(), frameNumber, lostPercent);
		}
		catch (...) {
			Utils::Log("PyroWave capture: write failed\n");
		}
	});
}

// Writes one decode unit to LocalState, verbatim, under a name that says what
// it holds: pyrowave_<W>x<H>_<chroma>_<depth>_<hdr|sdr>_f<frame>_<timestamp>.bin
// Automatic partial-frame captures insert "lost<NN>" before the timestamp: the
// percentage of the frame's transmitted blocks that never arrived.
//
// The file is exactly what the host framed and what the offline tools parse, so
// build_tools/pyrowave_dump_golden and tools/pyrowave_{loss,fec,tier}_sim.py all
// read it directly (partial captures need --allow-partial / --force-partial).
// The name is for humans only - every tool recovers the real geometry from the
// bitstream sequence header.
void PyroWaveDecoder::WriteCapture(const uint8_t *data, size_t length, int frameNumber, int lostPercent) {
	// LocalFolder->Path is stable for the process; resolving it per frame would
	// put WinRT calls on the decode thread for no reason.
	static std::string folder = [] {
		Platform::String ^ path = Windows::Storage::ApplicationData::Current->LocalFolder->Path;
		char buf[2048];
		if (wcstombs_s(NULL, buf, path->Data(), 2047) != 0)
			return std::string();
		return std::string(buf) + "\\";
	}();

	if (folder.empty()) {
		Utils::Log("PyroWave capture: could not resolve LocalState path\n");
		return;
	}

	bool chroma444 = (videoFormat & (VIDEO_FORMAT_PYROWAVE_444 | VIDEO_FORMAT_PYROWAVE10_444)) != 0;
	bool tenBit = (videoFormat & VIDEO_FORMAT_MASK_10BIT) != 0;
	bool hdr = tenBit && LiGetCurrentHostDisplayHdrMode();

	SYSTEMTIME now;
	GetLocalTime(&now);

	char lostTag[16] = "";
	if (lostPercent >= 0)
		sprintf_s(lostTag, "lost%02d_", lostPercent);

	char name[2048];
	sprintf_s(name, "%spyrowave_%dx%d_%s_%s_%s_f%05d_%s%04d%02d%02d-%02d%02d%02d.bin",
	          folder.c_str(), width, height,
	          chroma444 ? "444" : "420",
	          tenBit ? "10bit" : "8bit",
	          hdr ? "hdr" : "sdr",
	          frameNumber, lostTag,
	          now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond);

	FILE *f = nullptr;
	if (fopen_s(&f, name, "wb") != 0 || !f) {
		Utils::Logf("PyroWave capture: could not open %s\n", name);
		return;
	}
	size_t written = fwrite(data, 1, length, f);
	fclose(f);

	if (written != length)
		Utils::Logf("PyroWave capture: short write to %s (%zu of %zu bytes)\n", name, written, length);
	else
		Utils::Logf("PyroWave capture: wrote %s (%zu bytes)\n", name, length);
}

// Called by the VideoDec thread (CAPABILITY_DIRECT_SUBMIT)
int PyroWaveDecoder::SubmitDecodeUnit(PDECODE_UNIT decodeUnit) {
	LARGE_INTEGER decodeStart, decodeEnd;
	QueryPerformanceCounter(&decodeStart);
	if (m_StreamEpochQpc == 0)
		m_StreamEpochQpc = decodeStart.QuadPart;

	if (!m_decoder || !m_pool)
		return DR_OK;

	// Build a segment view of the decode unit. Nothing is copied: the decoder
	// parses the packet stream in place and Decode() uploads these same bytes
	// straight into the payload buffer.
	m_segments.clear();
	size_t length = 0;
	for (PLENTRY entry = decodeUnit->bufferList; entry != NULL; entry = entry->next) {
		PyroWaveD3D11::PayloadSegment segment;
		segment.data = reinterpret_cast<const uint8_t *>(entry->data);
		segment.size = (uint32_t)entry->length;
		segment.offset = (uint32_t)length;
		m_segments.push_back(segment);
		length += entry->length;
	}

	uint32_t droppedFramesNetwork = 0;
	if (m_LastFrameNumber > 0 && decodeUnit->frameNumber > (m_LastFrameNumber + 1)) {
		droppedFramesNetwork = decodeUnit->frameNumber - (m_LastFrameNumber + 1);
		// The bitstream sequence counter is only 3 bits: a burst of >=4 lost
		// frames can alias into the decoder's stale-packet window and stall
		// it for several more frames (or merge two frames at diff==0). We
		// know the exact gap from frameNumber, so force a full sequence
		// reset and let this frame's sequence be adopted unconditionally.
		m_decoder->Clear();
	}
	m_LastFrameNumber = decodeUnit->frameNumber;

	if (!decodeUnit->rtpTimestamp) {
		// Estimate for hosts that don't send timestamps (e.g. Wolf)
		LogOnce("Warning: host is not sending RTP timestamps, this may hurt frame pacing\n");
		double ptsMs = QpcToMs(QpcNow() - m_StreamEpochQpc);
		decodeUnit->rtpTimestamp = (uint32_t)(ptsMs * 90.0);
		decodeUnit->presentationTimeUs = (uint64_t)(ptsMs * 1000.0);
	}

	Stats::instance().SubmitVideoBytesAndReassemblyTime((int)length, decodeUnit, droppedFramesNetwork);

	// A partial DU is the contiguous prefix of a frame whose tail was lost.
	// Everything present is valid, it just stops at an arbitrary byte, so the
	// framing walk has to tolerate a truncated chunk instead of trusting the
	// count and sizes to reach the end of the buffer.
	bool partial = decodeUnit->isPartial;

	// Dev-tool capture. The manual (quick menu) capture only saves complete
	// frames - a partial DU stops at an arbitrary byte and would be useless as
	// golden-vector test data. The write goes to a background task: a few
	// hundred KB of console storage I/O on the decode thread would stall it
	// long enough to drop frames, and the resulting loss would change what the
	// rest of the burst captures.
	if (!partial && m_captureFramesRemaining.load(std::memory_order_relaxed) > 0) {
		m_captureFramesRemaining.fetch_sub(1, std::memory_order_relaxed);
		WriteCaptureAsync(decodeUnit->frameNumber, -1);
	}

	// Strip the transport framing: [u32 count]{[u32 size][bytes]}*. When the
	// active-block sideband is negotiated, [u32 mask_words][mask_words * u32]
	// follows the count. Each sized chunk holds one or more whole self-delimiting
	// pyrowave packets; PushPacket parses whatever it is handed, in place.
	if (length < 4) {
		Utils::Log("PyroWave live: runt decode unit\n");
		return DR_OK;
	}
	SegmentCursor cursor { m_segments.data(), m_segments.size(), 0, 0, length };
	uint32_t chunkCount;
	if (!cursor.ReadU32(chunkCount)) {
		Utils::Log("PyroWave live: truncated DU framing\n");
		return DR_OK;
	}

	// Mask of transmitted blocks in the FEC-protected bands. It rides inside
	// the host's FEC-protected head, so only a lost first packet or unrecovered
	// head loss keeps it from arriving; a zero word count means the host could
	// not compute it, which leaves the decoder on its conservative path.
	m_activeMask.clear();
	uint32_t maskWords = 0;
	if (m_activeBlockSideband) {
		if (!cursor.ReadU32(maskWords) || maskWords > kMaxActiveMaskWords ||
		    maskWords > cursor.remaining / sizeof(uint32_t)) {
			Utils::Log("PyroWave live: malformed active-block mask header\n");
			m_decoder->Clear();
			return DR_OK;
		}
		m_activeMask.resize(maskWords);
		for (uint32_t i = 0; i < maskWords; i++) {
			if (!cursor.ReadU32(m_activeMask[i])) {
				Utils::Log("PyroWave live: truncated active-block mask\n");
				m_activeMask.clear();
				m_decoder->Clear();
				return DR_OK;
			}
		}
	}

	for (uint32_t i = 0; i < chunkCount; i++) {
		if (cursor.remaining < 4)
			break;
		uint32_t sz;
		if (!cursor.ReadU32(sz))
			break;
		if (sz > cursor.remaining) {
			// Truncated final chunk. It can still hold whole pyrowave packets
			// ahead of the cut, so decode as far as the data goes.
			if (partial)
				m_decoder->PushPacket(m_segments.data(), m_segments.size(),
				                      cursor.index, cursor.offset, cursor.remaining, true);
			else
				Utils::Logf("PyroWave live: chunk %u overruns DU (frame %d)\n", i, decodeUnit->frameNumber);
			break;
		}
		m_decoder->PushPacket(m_segments.data(), m_segments.size(),
		                      cursor.index, cursor.offset, sz);
		cursor.Skip(sz);
	}

	// With the sideband, the explicit per-band check is authoritative: the mask
	// tells a block the encoder never transmitted from a lost one, so no
	// ordered-stream fallback is needed. With no mask (not negotiated, or the
	// host reported a zero word count), fall back to the inference-based
	// overload.
	bool ready = m_activeBlockSideband && maskWords > 0
	    ? m_decoder->DecodeIsReady(partial, PyroWaveD3D11::Decoder::kPartialPristineBands,
	                               PyroWaveD3D11::Decoder::kPartialMinimumPacketRatio,
	                               m_activeMask.data(), m_activeMask.size())
	    : m_decoder->DecodeIsReady(partial);
	if (!ready) {
		// For a complete DU this should not happen. For a partial one it just
		// means too little of the frame survived to be worth showing (the
		// decoder requires the coarse wavelet levels, 4 and 3, to be fully
		// present), so we keep the previous frame on screen instead of
		// flashing a wrecked one.
		if (!partial)
			Utils::Logf("PyroWave live: frame %d not decodable, dropping\n", decodeUnit->frameNumber);
		m_decoder->Clear();
		return DR_OK;
	}

	// Auto-save the first few partial frames that made it past the quality
	// floor (i.e. the ones that will actually render), so how they looked can
	// be reviewed offline after the session. The filename records how much of
	// the frame was lost, measured in transmitted blocks: the sequence header
	// says how many the host sent, the decoder counted how many arrived.
	if (partial && m_partialCapturesRemaining.load(std::memory_order_relaxed) > 0) {
		m_partialCapturesRemaining.fetch_sub(1, std::memory_order_relaxed);
		int totalBlocks = m_decoder->TotalBlocksInSequence();
		int lostPercent = totalBlocks > 0
		    ? 100 - (100 * m_decoder->DecodedBlocks()) / totalBlocks
		    : 0;
		WriteCaptureAsync(decodeUnit->frameNumber, lostPercent);
	}

	PyroWaveD3D11::FrameSet *set = m_pool->Acquire();
	if (!set) {
		// All plane sets in flight (renderer far behind); drop this frame.
		Utils::Log("PyroWave live: frame pool exhausted, dropping frame\n");
		m_decoder->Clear();
		return DR_OK;
	}

	ID3D11UnorderedAccessView *uavs[3] = { set->uav[0].Get(), set->uav[1].Get(), set->uav[2].Get() };
	bool decoded;
	double gpuMs;
	bool haveGpuMs;
	{
		// The immediate context is shared with the render thread
		auto guard = FFMpegDecoder::Lock();
		auto *ctx = m_deviceResources->GetD3DDeviceContext();
		decoded = m_decoder->Decode(ctx, uavs, m_segments.data(), m_segments.size(), length);
		// Non-blocking: retrieves the measurement of a frame decoded a few
		// frames ago, if the GPU has finished it.
		haveGpuMs = m_decoder->PollGpuTimeMs(ctx, &gpuMs);
	}
	if (haveGpuMs)
		Stats::instance().SubmitGpuDecodeMs(gpuMs);
	if (!decoded) {
		Utils::Log("PyroWave live: Decode failed\n");
		AVFrame *tmp = m_pool->WrapFrame(set); // recycle the set via the free callback
		if (tmp)
			av_frame_free(&tmp);
		return DR_OK;
	}

	AVFrame *frame = m_pool->WrapFrame(set);
	if (!frame)
		return DR_OK;
	frame->pts = (int64_t)decodeUnit->rtpTimestamp;
	ApplyFrameColor(frame);

	QueryPerformanceCounter(&decodeEnd);
	{
		// Attach pacing userdata (same contract as the FFmpeg path)
		AVBufferRef *buf = av_buffer_allocz(sizeof(MLFrameData));
		if (buf) {
			((MLFrameData *)buf->data)->decodeEndQpc = decodeEnd.QuadPart;
			frame->opaque_ref = buf;
		}
	}

	// Frame is owned by Pacer from here (drops/errors recycle via free cb)
	Pacer::instance().submitFrame(frame);

	Stats::instance().SubmitDecodeMs(QpcToMs(decodeEnd.QuadPart - decodeStart.QuadPart));
	return DR_OK;
}

} // namespace moonlight_xbox_dx
