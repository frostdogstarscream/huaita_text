(function () {
  function $(id) {
    return document.getElementById(id);
  }

  async function jsonFetch(url, options) {
    var response = await fetch(url, options || {});
    var data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.message || response.statusText);
    }
    return data;
  }

  function setText(id, text) {
    var node = $(id);
    if (node) node.textContent = text == null ? "" : String(text);
  }

  function setFallbackLink(id, href) {
    var node = $(id);
    if (!node) return;
    var url = href || "";
    if (node.tagName && node.tagName.toLowerCase() === "a") {
      node.href = url;
      node.textContent = url;
      return;
    }
    node.innerHTML = "";
    if (!url) return;
    var a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = url;
    node.appendChild(a);
  }

  function setCaptureEnabled(enabled) {
    var button = $("captureButton");
    if (button) button.disabled = !enabled;
  }

  function cameraReady(camera) {
    return Boolean(camera && camera.opened && camera.has_frame);
  }

  function formatCameraStatus(camera) {
    if (!camera) return "摄像头状态未知";
    if (camera.opened && camera.has_frame) return "摄像头已连接";
    return "摄像头未连接";
  }

  var HUAITA_IMAGE_UPLOAD_URL =
    "http://36.150.215.5:48110/huaita_health/business/eos/image/upload";

  function qrcodeBase64ToDataUrl(raw) {
    if (!raw) return "";
    if (String(raw).indexOf("data:") === 0) return raw;
    return "data:image/png;base64," + raw;
  }

  async function loadUiConfig() {
    try {
      return await jsonFetch("/api/ui-config");
    } catch (_) {
      return { kiosk_idle_return_seconds: 30 };
    }
  }

  function createIdleReturnController(options) {
    var seconds = Math.max(Number(options.seconds) || 30, 5);
    var onTick = options.onTick || function () {};
    var onTimeout = options.onTimeout || function () {};
    var events = ["pointerdown", "pointermove", "keydown", "touchstart"];
    var timer = null;
    var deadline = 0;

    function tick() {
      var remaining = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
      onTick(remaining);
      if (remaining <= 0) {
        stop();
        onTimeout();
      }
    }

    function reset() {
      deadline = Date.now() + seconds * 1000;
      tick();
    }

    function stop() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
      }
      events.forEach(function (name) {
        window.removeEventListener(name, reset);
      });
    }

    events.forEach(function (name) {
      window.addEventListener(name, reset, { passive: true });
    });
    timer = window.setInterval(tick, 250);
    reset();

    return { reset: reset, stop: stop };
  }

  function initCameraPreview() {
    var img = $("cameraStream");
    var fallback = $("cameraFallback");
    var fallbackText = $("cameraFallbackText");
    var mode = $("previewMode");
    var fitToggle = $("previewFitToggle");
    if (!img) return function () {};

    var stopped = false;
    var frameTimer = null;
    var watchdog = null;
    var objectUrl = null;
    var fitStorageKey = "huaita.preview.fitMode";
    var fitMode = "contain";

    function applyFitMode(nextMode) {
      fitMode = nextMode === "cover" ? "cover" : "contain";
      img.setAttribute("data-fit-mode", fitMode);
      if (fitToggle) {
        fitToggle.textContent = fitMode === "contain" ? "切换为铺满" : "切换为完整";
        fitToggle.setAttribute("aria-pressed", fitMode === "cover" ? "true" : "false");
      }
      try {
        window.localStorage.setItem(fitStorageKey, fitMode);
      } catch (_) {}
    }

    function clearTimers() {
      if (frameTimer) window.clearInterval(frameTimer);
      if (watchdog) window.clearTimeout(watchdog);
      frameTimer = null;
      watchdog = null;
    }

    function revokeUrl() {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
      }
    }

    function setMode(text) {
      if (mode) mode.textContent = text;
    }

    function showFallback(text) {
      if (fallback) fallback.style.display = "";
      if (fallbackText) fallbackText.textContent = text;
    }

    function hideFallback() {
      if (fallback) fallback.style.display = "none";
    }

    async function pullFrame() {
      if (stopped) return;
      try {
        var r = await fetch("/api/camera/frame?_ts=" + Date.now(), { cache: "no-store" });
        if (!r.ok) throw new Error("frame unavailable");
        var blob = await r.blob();
        revokeUrl();
        objectUrl = URL.createObjectURL(blob);
        img.src = objectUrl;
        setMode("实时预览（单帧轮询）");
        hideFallback();
      } catch (_) {
        showFallback("实时画面暂不可用，请检查摄像头连接");
      }
    }

    function startFramePolling(reasonText) {
      clearTimers();
      setMode("实时预览（单帧轮询）");
      showFallback(reasonText || "视频流不可用，已切换轮询模式");
      pullFrame();
      frameTimer = window.setInterval(pullFrame, 250);
    }

    function startStream() {
      clearTimers();
      setMode("实时预览（视频流）");
      showFallback("正在连接视频流…");
      img.src = "/video_feed?_ts=" + Date.now();
      img.onload = function () {
        if (!stopped) hideFallback();
      };
      img.onerror = function () {
        if (!stopped) startFramePolling("视频流连接失败，已切换轮询模式");
      };
      watchdog = window.setTimeout(function () {
        if (!stopped && (!img.complete || !img.naturalWidth)) {
          startFramePolling("视频流超时，已切换轮询模式");
        }
      }, 2200);
    }

    try {
      var savedFitMode = window.localStorage.getItem(fitStorageKey);
      if (savedFitMode === "cover" || savedFitMode === "contain") {
        fitMode = savedFitMode;
      }
    } catch (_) {}
    applyFitMode(fitMode);
    if (fitToggle) {
      fitToggle.addEventListener("click", function () {
        applyFitMode(fitMode === "contain" ? "cover" : "contain");
      });
    }

    async function freezeFrame() {
      if (stopped) return;
      clearTimers();
      try {
        var r = await fetch("/api/camera/frame?_ts=" + Date.now(), { cache: "no-store" });
        if (r.ok) {
          var blob = await r.blob();
          revokeUrl();
          objectUrl = URL.createObjectURL(blob);
          img.src = objectUrl;
          hideFallback();
          setMode("画面已冻结");
          return;
        }
      } catch (_) {}
      img.removeAttribute("src");
      showFallback("画面已冻结");
      setMode("画面已冻结");
    }

    startStream();

    function cleanup() {
      stopped = true;
      clearTimers();
      revokeUrl();
      img.removeAttribute("src");
    }

    cleanup.freezeFrame = freezeFrame;
    return cleanup;
  }

  async function pollWelcomeState() {
    var state = window.__huaitaWelcomeState || {};

    try {
      var template = await jsonFetch("/api/current-template");
      setText("currentSlogan", template.slogan);
      setText("countdownText", template.seconds_to_next + " 秒后轮播下一句");
    } catch (_) {
      setText("currentSlogan", "标语读取失败");
      setText("countdownText", "请稍后重试");
    }

    try {
      var laser = await Promise.race([
        jsonFetch("/api/laser-status"),
        new Promise(function (_, reject) {
          window.setTimeout(function () {
            reject(new Error("timeout"));
          }, 1200);
        }),
      ]);
      setText("laserText", laser.message || laser.trigger_state || "手动拍照模式");
    } catch (_) {
      setText("laserText", "手动拍照模式");
    }

    try {
      var health = await jsonFetch("/api/health");
      state.lastCameraReady = cameraReady(health.camera);
      setText("cameraText", formatCameraStatus(health.camera));
      if (state.autoTaskInProgress || state.manualCaptureInProgress) {
        setCaptureEnabled(false);
      } else {
        setCaptureEnabled(state.lastCameraReady);
      }
      if (!state.autoTaskInProgress && !state.manualCaptureInProgress) {
        setText("captureStatus", state.lastCameraReady ? "等待拍照" : "摄像头未就绪");
      }
    } catch (_) {
      setText("cameraText", "摄像头状态读取失败");
      setCaptureEnabled(false);
    }
  }

  function pollTask(taskId) {
    var statusNode = $("captureStatus");
    var button = $("captureButton");
    var state = window.__huaitaWelcomeState || {};

    async function tick() {
      try {
        var task = await jsonFetch("/api/task/" + encodeURIComponent(taskId));
        if (statusNode) statusNode.textContent = task.message || task.status;
        if (task.status === "completed") {
          state.manualCaptureInProgress = false;
          sessionStorage.setItem("huaihaiLastTask", JSON.stringify(task));
          location.href = "select.html?task_id=" + encodeURIComponent(taskId);
          return;
        }
        if (task.status === "failed") {
          state.manualCaptureInProgress = false;
          if (button) button.disabled = false;
          return;
        }
        window.setTimeout(tick, 1200);
      } catch (error) {
        state.manualCaptureInProgress = false;
        if (statusNode) statusNode.textContent = "任务查询失败: " + error.message;
        if (button) button.disabled = false;
      }
    }
    tick();
  }

  async function handleCapture() {
    var statusNode = $("captureStatus");
    var button = $("captureButton");
    var state = window.__huaitaWelcomeState || {};
    if (button) button.disabled = true;
    if (statusNode) statusNode.textContent = "正在创建拍照任务…";
    state.manualCaptureInProgress = true;
    try {
      var task = await jsonFetch("/api/capture", { method: "POST" });
      pollTask(task.task_id);
    } catch (error) {
      state.manualCaptureInProgress = false;
      if (statusNode) statusNode.textContent = "拍照失败: " + error.message;
      if (button) button.disabled = false;
    }
  }

  async function handleSync() {
    var syncSequence = $("syncSequence");
    var sequenceNo = Number(syncSequence ? syncSequence.value : "1");
    if (!Number.isInteger(sequenceNo) || sequenceNo < 1 || sequenceNo > 75) {
      setText("captureStatus", "序号请输入 1 到 75 之间的整数");
      return;
    }
    try {
      var result = await jsonFetch("/api/sync-time", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence_no: sequenceNo }),
      });
      if (syncSequence && result.sequence_no) syncSequence.value = String(result.sequence_no);
      await pollWelcomeState();
      setText("captureStatus", "已同步到第 " + result.sequence_no + " 句");
    } catch (error) {
      setText("captureStatus", "同步失败: " + error.message);
    }
  }

  async function initWelcomePage() {
    window.__huaitaWelcomeState = {
      manualCaptureInProgress: false,
      autoTaskInProgress: false,
      lastCameraReady: false,
    };
    var cleanupPreview = initCameraPreview();
    var captureButton = $("captureButton");
    var syncButton = $("syncButton");
    if (captureButton) captureButton.addEventListener("click", handleCapture);
    if (syncButton) syncButton.addEventListener("click", handleSync);
    await pollWelcomeState();
    var timer = window.setInterval(pollWelcomeState, 1000);
    window.addEventListener(
      "pagehide",
      function () {
        window.clearInterval(timer);
        cleanupPreview();
      },
      { once: true }
    );
  }

  function initKioskWaitPage() {
    var pollTimer = null;
    var navigating = false;
    var consecutive = 0;
    var need = 3;
    var pollMs = 200;

    async function tick() {
      if (navigating) return;
      try {
        var s = await jsonFetch("/api/laser-status");
        if (s && s.enabled && s.person_in_range) {
          consecutive += 1;
          if (consecutive >= need) {
            navigating = true;
            if (pollTimer) {
              window.clearInterval(pollTimer);
              pollTimer = null;
            }
            window.location.href = "/camera.html";
            return;
          }
        } else {
          consecutive = 0;
        }
      } catch (_) {
        consecutive = 0;
      }
    }

    pollTimer = window.setInterval(tick, pollMs);
    tick();

    window.addEventListener(
      "pagehide",
      function () {
        if (pollTimer) {
          window.clearInterval(pollTimer);
          pollTimer = null;
        }
      },
      { once: true }
    );
  }

  async function initCameraFullscreenPage() {
    var cleanupPreview = initCameraPreview();
    var countdownNode = $("countdownNumber");
    var titleNode = $("cameraPromptTitle");
    var subtitleNode = $("cameraPromptText");
    var transitionMask = $("cameraTransitionMask");
    var transitionMaskTitle = transitionMask ? transitionMask.querySelector(".camera-transition-mask__title") : null;
    var transitionMaskSubtitle = transitionMask ? transitionMask.querySelector(".camera-transition-mask__subtitle") : null;
    var heartbeatTimer = null;
    var stopped = false;
    var enterTime = Math.floor(Date.now() / 1000);
    var lastHandledTaskId = "";
    var redirecting = false;
    var pollTimer = null;
    var transitionTaskId = "";
    var transitionPhase = "idle";
    var taskSeenAt = Object.create(null);
    var laserVacantTicks = 0;
    var LASER_VACANT_TICKS_NEED = 8;
    var noPersonReturning = false;

    async function reportCameraPageActive(active) {
      try {
        await fetch("/api/camera-page-active", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ active: Boolean(active) }),
          keepalive: true,
        });
      } catch (_) {}
    }

    function isLaserTask(task) {
      return Boolean(task && typeof task === "object" && task.trigger_source === "laser" && task.task_id);
    }

    function isTaskFromCurrentSession(task) {
      if (!isLaserTask(task)) return false;
      var capturedAt = Number(task.captured_at) || 0;
      if (capturedAt > 0) return capturedAt >= enterTime;
      var taskId = String(task.task_id || "");
      if (!taskSeenAt[taskId]) {
        taskSeenAt[taskId] = Math.floor(Date.now() / 1000);
      }
      return taskSeenAt[taskId] >= enterTime;
    }

    function setTransitionMaskVisible(visible) {
      if (!transitionMask) return;
      if (visible) {
        transitionMask.classList.add("is-visible");
        transitionMask.setAttribute("aria-hidden", "false");
      } else {
        transitionMask.classList.remove("is-visible");
        transitionMask.setAttribute("aria-hidden", "true");
      }
    }

    function enterWorkingTransition(taskId) {
      transitionTaskId = taskId;
      transitionPhase = "working";
      document.body.classList.add("is-transitioning-to-select");
      setTransitionMaskVisible(true);
      if (typeof cleanupPreview.freezeFrame === "function") {
        cleanupPreview.freezeFrame();
      }
      if (transitionMaskTitle) transitionMaskTitle.textContent = "照片正在生成中";
      if (transitionMaskSubtitle) transitionMaskSubtitle.textContent = "";
      if (titleNode) titleNode.textContent = "";
      if (subtitleNode) subtitleNode.textContent = "";
      if (countdownNode) countdownNode.textContent = "";
      setText("captureStatus", "");
      console.debug("[camera-transition]", "working", taskId);
    }

    function clearTransitionState() {
      transitionPhase = "idle";
      transitionTaskId = "";
      document.body.classList.remove("is-transitioning-to-select");
      setTransitionMaskVisible(false);
    }

    async function showNoPersonThenReturnWait(message) {
      if (redirecting || stopped || noPersonReturning) return;
      noPersonReturning = true;
      var tip = message || "未检测到人物";
      document.body.classList.add("is-transitioning-to-select");
      setTransitionMaskVisible(true);
      if (transitionMaskTitle) transitionMaskTitle.textContent = tip;
      if (transitionMaskSubtitle) transitionMaskSubtitle.textContent = "3 秒后返回等待页面";
      if (titleNode) titleNode.textContent = "";
      if (subtitleNode) subtitleNode.textContent = "";
      if (countdownNode) countdownNode.textContent = "";
      setText("captureStatus", "");
      await new Promise(function (resolve) {
        window.setTimeout(resolve, 3000);
      });
      await returnToKioskWaitOnVacancy();
    }

    function shouldRedirectToSelect(task) {
      if (!task || typeof task !== "object") return false;
      if (task.status !== "completed") return false;
      if (!isLaserTask(task)) return false;
      if (!Array.isArray(task.results) || task.results.length <= 0) return false;
      if (!isTaskFromCurrentSession(task)) return false;
      var taskId = String(task.task_id || "");
      if (!taskId || taskId === lastHandledTaskId) return false;
      return true;
    }

    async function syncLaserTaskTransitionByLatestTask() {
      if (redirecting || stopped) return;
      try {
        var payload = await jsonFetch("/api/latest-task");
        var task = payload && payload.task;
        if (!isLaserTask(task)) return;
        var taskId = String(task.task_id || "");
        var status = String(task.status || "");
        if (!isTaskFromCurrentSession(task) || taskId === lastHandledTaskId) return;
        if ((status === "queued" || status === "processing") && transitionPhase !== "done") {
          if (transitionPhase !== "working" || transitionTaskId !== taskId) {
            enterWorkingTransition(taskId);
          }
          return;
        }
        if (status === "failed") {
          if (transitionTaskId === taskId) {
            var failMessage = String(task.message || "");
            if (failMessage.indexOf("未检测到人物") >= 0) {
              await showNoPersonThenReturnWait("未检测到人物");
              return;
            }
            clearTransitionState();
            if (titleNode) titleNode.textContent = "快速准备好";
            if (subtitleNode) subtitleNode.textContent = "倒计时结束立即拍照";
            setText("captureStatus", task.message || "生成失败，请重试");
          }
          return;
        }
        if (!shouldRedirectToSelect(task)) return;
        if (transitionTaskId && transitionTaskId !== taskId) return;
        redirecting = true;
        transitionTaskId = taskId;
        transitionPhase = "done";
        lastHandledTaskId = taskId;
        if (pollTimer) {
          window.clearInterval(pollTimer);
          pollTimer = null;
        }
        document.body.classList.add("is-transitioning-to-select");
        setTransitionMaskVisible(true);
        if (transitionMaskTitle) transitionMaskTitle.textContent = "照片生成完成";
        if (transitionMaskSubtitle) transitionMaskSubtitle.textContent = "正在进入选图…";
        if (titleNode) titleNode.textContent = "";
        if (subtitleNode) subtitleNode.textContent = "";
        if (countdownNode) countdownNode.textContent = "";
        setText("captureStatus", "");
        console.debug("[camera-transition]", "done", taskId);
        await new Promise(function (resolve) {
          window.setTimeout(resolve, 1100);
        });
        await cleanupCameraFullscreen();
        var target = new URL("select.html", window.location.href);
        target.searchParams.set("task_id", lastHandledTaskId);
        window.location.href = target.toString();
      } catch (_) {}
    }

    async function returnToKioskWaitOnVacancy() {
      if (redirecting || stopped) return;
      redirecting = true;
      await cleanupCameraFullscreen();
      window.location.href = "/kiosk-wait.html";
    }

    async function pollFullscreenState() {
      if (redirecting || stopped || transitionPhase !== "idle") return;
      try {
        var laser = await jsonFetch("/api/laser-status");

        /* 激光已启用且已连接：连续检测到无人则回待机页 */
        var monitorVacancy =
          laser &&
          laser.enabled === true &&
          laser.connected === true &&
          typeof laser.person_in_range === "boolean";
        if (monitorVacancy) {
          if (laser.person_in_range === false) {
            laserVacantTicks += 1;
            if (laserVacantTicks >= LASER_VACANT_TICKS_NEED) {
              laserVacantTicks = 0;
              await returnToKioskWaitOnVacancy();
              return;
            }
          } else {
            laserVacantTicks = 0;
          }
        } else {
          laserVacantTicks = 0;
        }

        if (laser && laser.camera_page_active === false) {
          setText("captureStatus", "页面未激活，激光触发已暂停");
        }
        if (laser && laser.countdown_active) {
          var remain = Math.max(0, Number(laser.countdown_remaining) || 0);
          if (countdownNode) countdownNode.textContent = remain > 0 ? String(remain) : "";
          if (titleNode) titleNode.textContent = "快速准备好";
          if (subtitleNode) subtitleNode.textContent = "倒计时结束立即拍照";
          setText("captureStatus", remain > 0 ? "即将拍照" : "即将拍照");
        } else {
          if (countdownNode) countdownNode.textContent = "";
          if (titleNode) titleNode.textContent = "快速准备好";
          if (subtitleNode) subtitleNode.textContent = "倒计时结束立即拍照";
          setText("captureStatus", "等待拍照");
        }
      } catch (_) {
        laserVacantTicks = 0;
        if (countdownNode) countdownNode.textContent = "";
        setText("captureStatus", "状态读取中");
      }
    }

    function startHeartbeat() {
      if (heartbeatTimer) {
        window.clearInterval(heartbeatTimer);
      }
      heartbeatTimer = window.setInterval(function () {
        if (!stopped && document.visibilityState === "visible") {
          reportCameraPageActive(true);
        }
      }, 1000);
    }

    function stopHeartbeat() {
      if (heartbeatTimer) {
        window.clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    }

    async function cleanupCameraFullscreen() {
      if (stopped) return;
      stopped = true;
      clearTransitionState();
      stopHeartbeat();
      if (pollTimer) {
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
      await reportCameraPageActive(false);
      cleanupPreview();
    }

    await reportCameraPageActive(true);
    startHeartbeat();
    await pollFullscreenState();
    pollTimer = window.setInterval(function () {
      pollFullscreenState();
      syncLaserTaskTransitionByLatestTask();
    }, 500);

    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "visible") {
        reportCameraPageActive(true);
      } else {
        reportCameraPageActive(false);
      }
    });

    window.addEventListener("beforeunload", function () {
      reportCameraPageActive(false);
    });

    window.addEventListener(
      "pagehide",
      function () {
        if (pollTimer) {
          window.clearInterval(pollTimer);
          pollTimer = null;
        }
        cleanupCameraFullscreen();
      },
      { once: true }
    );
  }

  function viewHref(item) {
    var u = new URL("view.html", window.location.href);
    u.searchParams.set("src", item.image_url);
    u.searchParams.set("image_id", item.image_id || "");
    u.searchParams.set("background_id", item.background_id || "");
    u.searchParams.set("background_name", item.background_name || "");
    return u.toString();
  }

  function isRenderableTask(task) {
    return Boolean(
      task &&
        task.status === "completed" &&
        Array.isArray(task.results) &&
        task.results.length > 0
    );
  }

  function renderSelectTask(task, grid) {
    sessionStorage.setItem("huaihaiLastTask", JSON.stringify(task));
    setText("selectSlogan", task.slogan || "当前标语");
    setText("selectStatus", "请选择一张背景图");
    grid.innerHTML = "";
    task.results.forEach(function (item, index) {
      var link = document.createElement("a");
      link.className = "photo-frame";
      link.href = viewHref(item);
      link.innerHTML =
        '<span class="photo-frame__badge">' +
        (index + 1) +
        "</span>" +
        '<img src="' +
        item.image_url +
        '" alt="' +
        (item.background_name || "背景") +
        '">';
      grid.appendChild(link);
    });
  }

  function groupResultsByBackground(results) {
    var items = Array.isArray(results) ? results.slice() : [];
    if (!items.length) return [];
    items.sort(function (a, b) {
      return (Number(a && a.order) || 0) - (Number(b && b.order) || 0);
    });
    var map = Object.create(null);
    var groups = [];
    items.forEach(function (item) {
      var key = String((item && item.background_id) || "unknown");
      if (!map[key]) {
        map[key] = { key: key, background_name: item.background_name || "背景", items: [] };
        groups.push(map[key]);
      }
      map[key].items.push(item);
    });
    return groups;
  }

  function ensureFourItems(items) {
    var source = Array.isArray(items) ? items : [];
    if (!source.length) return [];
    var selected = source.slice(0, 4);
    while (selected.length < 4) {
      selected.push(source[selected.length % source.length]);
    }
    return selected;
  }

  function renderSelectGroup(task, grid, groups, groupIndex) {
    if (!Array.isArray(groups) || !groups.length) {
      renderSelectTask(task, grid);
      return;
    }
    var idx = ((Number(groupIndex) || 0) % groups.length + groups.length) % groups.length;
    var group = groups[idx];
    var items = ensureFourItems(group.items);
    sessionStorage.setItem("huaihaiLastTask", JSON.stringify(task));
    setText("selectSlogan", task.slogan || "当前标语");
    setText(
      "selectStatus",
      "当前背景：" + (group.background_name || "背景") + "（" + (idx + 1) + "/" + groups.length + "）"
    );
    grid.innerHTML = "";
    items.forEach(function (item, index) {
      var link = document.createElement("a");
      link.className = "photo-frame";
      link.href = viewHref(item);
      link.innerHTML =
        '<span class="photo-frame__badge">' +
        (index + 1) +
        "</span>" +
        '<img src="' +
        item.image_url +
        '" alt="' +
        (item.background_name || "背景") +
        '">';
      grid.appendChild(link);
    });
  }

  async function resolveSelectTask(taskIdFromQuery) {
    if (taskIdFromQuery) {
      try {
        var taskById = await jsonFetch("/api/task/" + encodeURIComponent(taskIdFromQuery));
        if (isRenderableTask(taskById)) return taskById;
      } catch (_) {}
    }

    try {
      var raw = sessionStorage.getItem("huaihaiLastTask");
      if (raw) {
        var cached = JSON.parse(raw);
        if (isRenderableTask(cached)) return cached;
      }
    } catch (_) {}

    try {
      var latestPayload = await jsonFetch("/api/latest-task");
      if (isRenderableTask(latestPayload && latestPayload.task)) return latestPayload.task;
    } catch (_) {}

    return null;
  }

  /** 選片結果網格：點擊進 view 前先播退場過渡（View Transitions API 可用時優先用） */
  function attachSelectPhotoToViewTransition(grid) {
    if (!grid || grid.dataset.huaihaiSelectTransitionAttached === "1") return;
    grid.dataset.huaihaiSelectTransitionAttached = "1";
    var toViewBusy = false;
    var EXIT_MS = 520;

    function resetSelectExitState() {
      toViewBusy = false;
      if (document && document.body) {
        document.body.classList.remove("page--select-exit");
      }
    }

    window.addEventListener("pageshow", resetSelectExitState);

    grid.addEventListener(
      "click",
      function (ev) {
        var t = ev.target;
        var a =
          t && typeof t.closest === "function"
            ? t.closest("a.photo-frame")
            : null;
        if (!a || !a.href || toViewBusy) return;
        var href = a.href;
        toViewBusy = true;
        ev.preventDefault();

        function navigateView() {
          window.location.href = href;
        }

        document.body.classList.add("page--select-exit");
        window.setTimeout(navigateView, EXIT_MS);
      },
      false
    );
  }

  async function initSelectPage() {
    if (document && document.body) {
      document.body.classList.remove("page--select-exit");
    }
    window.addEventListener("pageshow", function () {
      if (document && document.body) {
        document.body.classList.remove("page--select-exit");
      }
    });
    var params = new URLSearchParams(window.location.search);
    var taskId = params.get("task_id");
    var grid = $("photoGrid") || $("resultGrid");

    if (!grid) {
      setText("selectStatus", "页面结构异常：缺少结果容器");
      setText("selectActionHint", "请联系维护人员检查前端模板。");
      return;
    }

    attachSelectPhotoToViewTransition(grid);

    var task = await resolveSelectTask(taskId);
    if (!task) {
      setText("selectStatus", taskId ? "任务不可用或尚未完成" : "缺少任务参数且无可恢复任务");
      setText("selectActionHint", "请返回首页重新拍照，或稍后重试。");
      return;
    }
    var groups = groupResultsByBackground(task.results);
    if (groups.length <= 1) {
      renderSelectTask(task, grid);
      return;
    }
    renderSelectGroup(task, grid, groups, 0);
  }

  async function initViewPage() {
    var params = new URLSearchParams(window.location.search);
    var src = params.get("src");
    var imageId = params.get("image_id");
    var img = $("preview");
    var qr = $("qrImage");
    var tip = $("qrHint");
    var loader = $("viewLoadingOverlay");
    function setViewLoading(active) {
      if (!loader) return;
      loader.classList.toggle("is-active", Boolean(active));
      loader.setAttribute("aria-hidden", active ? "false" : "true");
    }
    setViewLoading(true);
    var uiConfig = await loadUiConfig();
    var idle = createIdleReturnController({
      seconds: uiConfig.kiosk_idle_return_seconds || 30,
      onTick: function (remaining) {
        setText("viewReturnText", remaining + " 秒后自动返回待机页面");
      },
      onTimeout: function () {
        location.href = "/kiosk-wait.html";
      },
    });

    if (!src) {
      if (tip) tip.textContent = "缺少展示地址，请返回选图";
      setViewLoading(false);
      return;
    }

    try {
      var previewImageBase64 = "";
      var qrcodeDataUrl = "";

      try {
        var response = await fetch(src, { credentials: "same-origin" });
        if (!response.ok) {
          throw new Error("图片读取失败");
        }
        var blob = await response.blob();
        previewImageBase64 = await new Promise(function (resolve, reject) {
          var reader = new FileReader();
          reader.onloadend = function () {
            resolve(reader.result || "");
          };
          reader.onerror = function () {
            reject(new Error("图片转 base64 失败"));
          };
          reader.readAsDataURL(blob);
        });
      } catch (error) {
        console.error("[initViewPage] 图片转 base64 异常:", error);
        previewImageBase64 = "";
      }

      if (previewImageBase64) {
        try {
          var contentType = "image/png";
          var typeMatch = /^data:([^;]+);/i.exec(previewImageBase64);
          if (typeMatch) {
            contentType = typeMatch[1];
          }

          var ext = "png";
          if (/jpe?g/i.test(contentType)) {
            ext = "jpg";
          } else if (/webp/i.test(contentType)) {
            ext = "webp";
          }

          var safeName = String(imageId || "photo").replace(/[^\w.-]+/g, "_");
          var fileName = safeName + "." + ext;

          var uploadResp = await fetch(HUAITA_IMAGE_UPLOAD_URL, {
            method: "POST",
            headers: {
              accept: "application/json",
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              base64: previewImageBase64,
              fileName: fileName,
              contentType: contentType,
            }),
          });

          var uploadJson;
          try {
            uploadJson = await uploadResp.json();
          } catch (parseErr) {
            console.error("[initViewPage] 上传接口响应解析异常:", parseErr);
            throw new Error("上传接口返回非 JSON");
          }

          if (!uploadResp.ok || uploadJson.code !== 200) {
            throw new Error(uploadJson.msg || uploadResp.statusText || "上传失败");
          }

          var rawQr = uploadJson.data && uploadJson.data.qrcodeBase64;
          if (!rawQr) {
            throw new Error("响应中缺少 qrcodeBase64");
          }

          qrcodeDataUrl = qrcodeBase64ToDataUrl(rawQr);
        } catch (error) {
          console.error("[initViewPage] 获取展示二维码流程异常:", error);
          qrcodeDataUrl = "";
        }
      } else {
        console.error("[initViewPage] 无有效图片 base64，已跳过后端上传与二维码");
      }

      sessionStorage.setItem("huaihaiViewSrc", src);
      var previewReady = Promise.resolve();
      if (img) {
        previewReady = new Promise(function (resolve) {
          if (img.complete && img.getAttribute("src") === src) {
            resolve();
            return;
          }
          img.addEventListener("load", resolve, { once: true });
          img.addEventListener("error", resolve, { once: true });
        });
        img.src = src;
      }
      var qrReady = Promise.resolve();
      if (qr) {
        if (qrcodeDataUrl) {
          qrReady = new Promise(function (resolve) {
            qr.addEventListener("load", resolve, { once: true });
            qr.addEventListener("error", resolve, { once: true });
          });
          qr.src = qrcodeDataUrl;
          if (tip) tip.textContent = "请使用手机扫码下载";
        } else {
          qr.removeAttribute("src");
          if (tip) {
            tip.textContent = "二维码暂不可用，请稍后重试或返回选图";
          }
        }
        qr.alt = "下载二维码";
        qr.addEventListener("error", function () {
          if (tip) tip.textContent = "二维码加载失败，请稍后重试或返回";
        });
      }
      await Promise.all([previewReady, qrReady]);
      setViewLoading(false);
    } catch (error) {
      console.error("[initViewPage] 页面初始化失败:", error);
      if (tip) {
        tip.textContent = "页面加载遇到问题，请点击返回重新选择";
      }
      setViewLoading(false);
    }

    window.addEventListener(
      "pagehide",
      function () {
        idle.stop();
        setViewLoading(false);
      },
      { once: true }
    );
  }

  function initDownloadPage() {
    var params = new URLSearchParams(location.search);
    var src = params.get("src") || sessionStorage.getItem("huaihaiViewSrc");
    var statusEl = $("status");
    var tipEl = $("tip");
    var btn = $("btnSave");
    var img = $("photo");

    if (!src) {
      setText("status", "缺少图片地址");
      setText("tip", "请返回扫码页重新打开下载页面");
      return;
    }

    function baseName() {
      var m = src.match(/([^/\\]+\.(jpe?g|png|webp|gif))$/i);
      return m ? m[1] : "huaihai-photo.jpg";
    }

    function triggerAnchorDownload(blob) {
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = baseName();
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.setTimeout(function () {
        URL.revokeObjectURL(url);
      }, 3000);
    }

    async function trySave() {
      if (statusEl) statusEl.textContent = "正在下载照片…";
      if (tipEl) tipEl.textContent = "";
      try {
        var response = await fetch(src, { credentials: "same-origin" });
        if (!response.ok) throw new Error("图片读取失败");
        var blob = await response.blob();
        triggerAnchorDownload(blob);
        if (statusEl) statusEl.textContent = "照片已发起下载";
        if (tipEl) tipEl.textContent = "如果没有弹出保存，请点击下方按钮重试";
        if (btn) btn.hidden = false;
      } catch (error) {
        if (statusEl) statusEl.textContent = "下载失败";
        if (tipEl) tipEl.textContent = error.message || "请重试";
        if (btn) btn.hidden = false;
      }
    }

    if (img) {
      img.src = src;
      img.addEventListener("load", trySave);
      img.addEventListener("error", function () {
        setText("status", "图片加载失败");
      });
    }
    if (btn) btn.addEventListener("click", trySave);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var page = document.body && document.body.getAttribute("data-page");
    if (page === "welcome") initWelcomePage();
    if (page === "kiosk-wait") initKioskWaitPage();
    if (page === "camera-fullscreen") initCameraFullscreenPage();
    if (page === "select") initSelectPage();
    if (page === "view") initViewPage();
    if (page === "download") initDownloadPage();
  });
})();
