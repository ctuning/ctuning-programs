/*
 * mad - MPEG audio decoder
 * Copyright (C) 2000-2001 Robert Leslie
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: audio_esd.c,v 1.2 2001/11/03 00:49:07 rob Exp $
 */

# ifdef HAVE_CONFIG_H
#  include "config.h"
# endif

# include "global.h"

# include <unistd.h>
# include <errno.h>
/* # include <esd.h> */

# include "mad.h"
# include "audio.h"

# if defined(WORDS_BIGENDIAN)
#  define audio_pcm_s16  audio_pcm_s16be
# else
#  define audio_pcm_s16  audio_pcm_s16le
# endif

static char const *host;
static int esd;
static audio_pcmfunc_t *audio_pcm;

static
int init(struct audio_init *init)
{
  return 0;
}

static
int config(struct audio_config *config)
{
  return 0;
}

static
int output(unsigned char const *ptr, unsigned int len)
{
  return 0;
}

static
int play(struct audio_play *play)
{
  return 0;
}

static
int stop(struct audio_stop *stop)
{
  return 0;
}

static
int finish(struct audio_finish *finish)
{
  return 0;
}

int audio_esd(union audio_control *control)
{
  return 0;
}
