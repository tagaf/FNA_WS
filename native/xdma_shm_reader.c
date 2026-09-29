/* Resident XDMA C2H reader.
 *
 * Replicates the vendor dma_from_device tool's device access EXACTLY --
 * that CLI is the one readback path proven not to freeze this SoC
 * (see NOTES.md #5; raw Python read()/readv() with O_RDONLY hard-froze it):
 *
 *   - open(dev, O_RDWR | O_TRUNC)           <- same flags as the vendor tool
 *     (its comment: O_TRUNC tells the driver to flush; Python used O_RDONLY)
 *   - posix_memalign(4096) bounce buffer    <- same allocator + alignment
 *   - chunked lseek+read loop, RW_MAX_SIZE  <- same loop, same chunk cap
 *
 * The only difference: instead of writing the bounce buffer to an output
 * file and exiting, it memcpy()s into a shared /dev/shm mapping and waits
 * for the next command on stdin. The DMA target is still the same kind of
 * aligned heap buffer; the shm copy is plain CPU memcpy after DMA completes.
 *
 * Protocol (line-oriented, stdin/stdout):
 *   "R <addr> <nbytes>\n"             -> DMA read, copy to shm offset 0
 *   "R <addr> <nbytes> <shmoff>\n"    -> ... to shm offset <shmoff>
 *        both reply "OK <nbytes>\n"
 *   "Q\n"                             -> exit 0
 *   any failure  -> reply "ERR <errno-msg>\n" (process keeps running)
 *
 * The <shmoff> form exists for streaming: the caller sizes the shm as several
 * segment slots and rotates through them, so the consumer can still be
 * writing slot k while the next DMA lands in slot k+1. Without it every read
 * had to wait for the previous buffer to be drained.
 *
 * Usage: xdma_shm_reader <device> <shm_file> <shm_bytes> [max_read_bytes]
 *
 * <shm_bytes> sizes the shared mapping (the caller's ring). [max_read_bytes]
 * sizes the DMA bounce buffer and caps one read; it defaults to <shm_bytes>.
 * They are separate because a streaming ring is gigabytes while a single read
 * is one segment slice -- allocating a bounce buffer the size of a 30 GB ring
 * would be absurd even lazily faulted.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define RW_MAX_SIZE 0x7ffff000ULL   /* same cap as vendor dma_utils.c */

static ssize_t read_to_buffer(int fd, char *buffer, uint64_t size,
                              uint64_t base)
{   /* verbatim logic from dma_utils.c read_to_buffer() */
    ssize_t rc;
    uint64_t count = 0;
    char *buf = buffer;
    off_t offset = base;

    while (count < size) {
        uint64_t bytes = size - count;
        if (bytes > RW_MAX_SIZE)
            bytes = RW_MAX_SIZE;
        /* vendor code guards this with `if (offset)` -- correct for its
         * one-shot CLI (fresh fd starts at 0) but WRONG for a resident
         * process, where the previous command leaves the fd elsewhere.
         * Always seek; lseek(fd,0) is the same driver path. */
        rc = lseek(fd, offset, SEEK_SET);
        if (rc != offset)
            return -EIO;
        rc = read(fd, buf, bytes);
        if (rc < 0)
            return -EIO;
        count += rc;
        if ((uint64_t)rc != bytes)
            break;              /* underflow: stop, report what we got */
        buf += bytes;
        offset += bytes;
    }
    return count;
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <device> <shm_file> <shm_bytes> "
                        "[max_read_bytes]\n", argv[0]);
        return 2;
    }
    const char *dev = argv[1], *shm_path = argv[2];
    uint64_t max_bytes = strtoull(argv[3], NULL, 0);
    uint64_t max_read = (argc == 5) ? strtoull(argv[4], NULL, 0) : max_bytes;
    if (max_read == 0 || max_read > max_bytes)
        max_read = max_bytes;

    /* same open flags as the vendor tool. XSR_TEST_RDONLY=1 exists ONLY so
     * the protocol can be tested against a regular file (which O_TRUNC would
     * wipe); it must never be set when reading the real device. */
    int oflags = getenv("XSR_TEST_RDONLY") ? O_RDONLY : (O_RDWR | O_TRUNC);
    int fpga_fd = open(dev, oflags);
    if (fpga_fd < 0) {
        fprintf(stderr, "ERR open %s: %s\n", dev, strerror(errno));
        return 1;
    }

    int shm_fd = open(shm_path, O_RDWR | O_CREAT, 0600);
    if (shm_fd < 0 || ftruncate(shm_fd, max_bytes) != 0) {
        fprintf(stderr, "ERR shm %s: %s\n", shm_path, strerror(errno));
        return 1;
    }
    char *shm = mmap(NULL, max_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                     shm_fd, 0);
    if (shm == MAP_FAILED) {
        fprintf(stderr, "ERR mmap: %s\n", strerror(errno));
        return 1;
    }

    /* same allocator + alignment as the vendor tool */
    char *allocated = NULL;
    /* sized to the largest SINGLE read, not to the ring */
    if (posix_memalign((void **)&allocated, 4096, max_read + 4096)) {
        fprintf(stderr, "ERR memalign: %s\n", strerror(errno));
        return 1;
    }
    char *buffer = allocated;

    /* line-buffered replies; unbuffered enough for a pipe */
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("READY %llu %llu\n", (unsigned long long)max_bytes,
           (unsigned long long)max_read);

    char line[128];
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'Q')
            break;
        unsigned long long addr, nbytes, shmoff = 0;
        int got_args = sscanf(line, "R %llu %llu %llu", &addr, &nbytes, &shmoff);
        if (got_args < 2) {
            printf("ERR bad command\n");
            continue;
        }
        if (got_args < 3)
            shmoff = 0;
        /* both halves of the bound check, and the sum, in unsigned 64-bit --
         * shmoff + nbytes cannot wrap for any value fgets can deliver. */
        if (nbytes > max_read) {
            printf("ERR read %llu > max_read %llu\n", nbytes,
                   (unsigned long long)max_read);
            continue;
        }
        if (shmoff > max_bytes || shmoff + nbytes > max_bytes) {
            printf("ERR size %llu at off %llu > shm %llu\n", nbytes, shmoff,
                   (unsigned long long)max_bytes);
            continue;
        }
        ssize_t got = read_to_buffer(fpga_fd, buffer, nbytes, addr);
        if (got < 0) {
            printf("ERR dma read: %s\n", strerror(-got));
            continue;
        }
        memcpy(shm + shmoff, buffer, got);
        /* make the bytes visible to the Python mmap before the reply */
        __sync_synchronize();
        printf("OK %zd\n", got);
    }
    free(allocated);
    close(fpga_fd);
    return 0;
}
