(function () {
    function getCookie(name) {
        const cookieString = document.cookie || "";
        if (!cookieString) {
            return "";
        }
        const pairs = cookieString.split("; ");
        for (const pair of pairs) {
            if (!pair) {
                continue;
            }
            const separator = pair.indexOf("=");
            const key = separator >= 0 ? pair.slice(0, separator) : pair;
            if (key === name) {
                const value = separator >= 0 ? pair.slice(separator + 1) : "";
                return decodeURIComponent(value);
            }
        }
        return "";
    }

    function getImagesFromCr(pageCount) {
        if (!window.__cr || typeof window.__cr.getPicUrl !== "function") {
            return [];
        }
        const urls = [];
        for (let i = 1; i <= pageCount; i += 1) {
            try {
                const url = window.__cr.getPicUrl(i);
                if (typeof url === "string" && /^(?:https?:)?\/\//i.test(url)) {
                    urls.push(url);
                }
            } catch (error) {
                console.warn("colamanga getPicUrl failed", error);
            }
        }
        return urls;
    }

    try {
        if (window.__cr) {
            window.__cr.isfromMangaRead = 1;
        }
        if (window.__cad && typeof window.__cad.setCookieValue === "function") {
            window.__cad.setCookieValue();
        }
        const pageId = window.mh_info && window.mh_info.pageid ? String(window.mh_info.pageid) : "";
        const cadValues = window.__cad && typeof window.__cad.getCookieValue === "function"
            ? window.__cad.getCookieValue()
            : ["", ""];
        const cookieKey = String(cadValues[1] || "") + pageId;
        let pageCount = parseInt(getCookie(cookieKey) || "0", 10);
        let urls = [];

        if (pageCount > 0) {
            urls = getImagesFromCr(pageCount);
        }

        if (!urls.length) {
            if (window.mh_info && Array.isArray(window.mh_info.images)) {
                urls = window.mh_info.images.map(String);
            } else {
                document.querySelectorAll("img[data-src], img[srcset], img[src]").forEach((img) => {
                    const src = img.getAttribute("data-src") || img.getAttribute("src") || "";
                    if (/^(?:https?:)?\/\//i.test(src)) {
                        urls.push(src);
                    }
                });
            }
        }

        urls = Array.from(new Set(urls));
        if (pageCount && urls.length > pageCount) {
            urls = urls.slice(0, pageCount);
        }

        return { ok: true, urls, pageCount: pageCount || null };
    } catch (error) {
        return { ok: false, error: String(error), urls: [] };
    }
})();
